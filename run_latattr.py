# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from common.config import register_args, load_config_and_tokenizer
from common.utils import mkdir_f
from common.custom_squad_feature import custom_squad_convert_examples_to_features, SquadResult, SquadProcessor
from common.qa_metrics import (compute_predictions_logits,hotpot_evaluate,)
from run_qa import load_and_cache_examples, set_seed, to_list
from expl_methods.latattr_models import LAtAttrRobertaForQuestionAnswering
from common.interp_utils import compute_predictions_index_and_logits, merge_predictions, remove_padding
from vis_tools.vis_utils import visualize_attention_attributions
from itertools import combinations

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import itertools

def dump_attention_interp_info(args, examples, features, tokenizer, predictions, prelim_results, attentions, attributions):
    
    # attentions, attributions
    # N_Layer * B * N_HEAD * L * L
    attentions = attentions.detach().cpu().requires_grad_(False)
    attentions = torch.transpose(attentions, 0, 1)
    attributions = attributions.detach().cpu().requires_grad_(False)
    attributions = torch.transpose(attributions, 0, 1)

    for example, feature, prelim_result, attention, attribution in zip(
        examples,
        features,
        prelim_results,
        torch.unbind(attentions),
        torch.unbind(attributions)
    ):
        actual_len = len(feature.tokens)
        attention = attention[:,:,:actual_len, :actual_len].clone().detach()
        attribution = attribution[:,:,:actual_len, :actual_len].clone().detach()
        filename = os.path.join(args.interp_dir, f'{feature.example_index}-{feature.qas_id}.bin')
        prelim_result = prelim_result._asdict()
        prediction = predictions[example.qas_id]
        torch.save({'example': example, 'feature': feature, 'prediction': prediction, 'prelim_result': prelim_result,
            'attention': attention, 'attribution': attribution}, filename)

def predict_and_layerwise_attribute(args, batch, model, tokenizer, batch_features, batch_examples):
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)

    num_layers = model.num_hidden_layers

    # run predictions
    with torch.no_grad():
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "output_attentions": True,
        }

        if args.model_type in ["roberta", "distilbert", "camembert", "bart"]:
            del inputs["token_type_ids"]

        feature_indices = batch[3]
        outputs = model.restricted_forward(**inputs)

    batch_start_logits, batch_end_logits, batch_attentions = outputs
    outputs = outputs[:-1]

    batch_results = []
    for i, feature_index in enumerate(feature_indices):
        eval_feature = batch_features[i]
        unique_id = int(eval_feature.unique_id)

        output = [to_list(output[i]) for output in outputs]
        start_logits, end_logits = output
        result = SquadResult(unique_id, start_logits, end_logits)

        batch_results.append(result)
    
    batch_prelim_results, batch_predictions = compute_predictions_index_and_logits(
        batch_examples,
        batch_features,
        batch_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        tokenizer,
        args.dataset,
    )

    # run attributions
    batch_start_indexes = torch.LongTensor([x.start_index for x in batch_prelim_results]).to(args.device)
    batch_end_indexes = torch.LongTensor([x.end_index for x in batch_prelim_results]).to(args.device)
    batch_attentions = torch.stack(batch_attentions)
    
    active_layers = [1 for _ in range(num_layers)]

    # for data parallel 
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
        "active_layers": active_layers,
        "input_attentions": batch_attentions,
        "start_indexes": batch_start_indexes,
        "end_indexes": batch_end_indexes,
        "final_start_logits": batch_start_logits,
        "final_end_logits": batch_end_logits,
        "num_steps": args.ig_steps,
    }
    if args.model_type in ["roberta", "distilbert", "camembert", "bart"]:
        del inputs["token_type_ids"]
    
    batch_attributions = model.layer_attribute(**inputs)
    # print(batch_attributions.size())
    # attribution in logits
    return batch_predictions, batch_prelim_results, batch_attentions, batch_attributions

def attention_interp(args, model, tokenizer, prefix=""):

    if not os.path.exists(args.interp_dir):
        os.makedirs(args.interp_dir)
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    
    # fix the model
    model.requires_grad_(False)
    # assume one on on mapping
    assert len(examples) == len(features)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # restrict evak batch size
    args.eval_batch_size = 1

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)


    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_predictions = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
       
        feature_indices = to_list(batch[3])
        batch_features = [features[i] for i in feature_indices]
        batch_examples = [examples[i] for i in feature_indices]
        # batch prem, batch predictions
        batch = remove_padding(batch, batch_features[0])
        batch_predictions, batch_prelim_results, batch_attentions, batch_attributions = predict_and_layerwise_attribute(
            args,
            batch,
            model,
            tokenizer,
            batch_features,
            batch_examples
        )

        # lots of info, dump to files immediately
        dump_attention_interp_info(args, batch_examples, batch_features, tokenizer, batch_predictions, batch_prelim_results, batch_attentions, batch_attributions)
        all_predictions.append(batch_predictions)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    # output_prediction_file =  os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    # output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    # Compute the F1 and exact scores.
    all_predictions = merge_predictions(all_predictions)
    results = hotpot_evaluate(examples[:len(all_predictions)], all_predictions)
    return results

    
def vis_analyze(args, tokenizer):
    filenames = os.listdir(args.interp_dir)
    filenames.sort(key=lambda x: int(x.split('-')[0]))
    # print(len(filenames))
    datset_stats = []
    mkdir_f(args.visual_dir)
    for fname in tqdm(filenames, desc='Visualizing'):
        interp_info = torch.load(os.path.join(args.interp_dir, fname))        
        visualize_attention_attributions(args, tokenizer, interp_info, do_head=args.vis_head, do_layer=args.vis_layer)

def main():
    parser = argparse.ArgumentParser()
    register_args(parser)

    parser.add_argument(
        "--per_gpu_eval_batch_size", default=10, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--ig_steps", type=int, default=300, help="steps for running integrated gradient")
    parser.add_argument("--do_vis", action="store_true", help="Whether to run vis on the dev set.")
    parser.add_argument("--interp_dir",default=None,type=str,required=True,help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--visual_dir",default=None,type=str,help="The output visualization dir.")
    parser.add_argument("--vis_layer",action="store_true", help="Whether to vis each layer.")
    parser.add_argument("--vis_head",action="store_true", help="Whether to vis each head.")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config, tokenizer = load_config_and_tokenizer(args)

    if args.do_vis:
        vis_analyze(args, tokenizer)
    else:
        # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
        logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
        checkpoint = args.model_name_or_path
        logger.info("Evaluate the following checkpoints: %s", checkpoint)

        # Reload the model
        model = LAtAttrRobertaForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
        model.to(args.device)

        # Evaluate
        result = attention_interp(args, model, tokenizer, prefix="")
        logger.info("Result: {}".format(result))

if __name__ == "__main__":
    main()