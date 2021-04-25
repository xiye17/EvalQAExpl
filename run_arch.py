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
from common.dataset_utils import merge_tokens_into_words
from transformers.modeling_roberta import create_position_ids_from_input_ids
from vis_tools.vis_utils import visualize_attention_attributions
from itertools import combinations

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import itertools
from functools import reduce

def masked_token_prob(tokenizer, model, base_inputs, full_input_ids, *segments):
    input_ids = full_input_ids.clone()
    for c_range in segments:
        input_ids[0, c_range[0]:c_range[1]] = tokenizer.pad_token_id
    prob = model.probe_forward(**base_inputs, input_ids=input_ids)
    return prob

def kept_token_prob(tokenizer, model, base_inputs, full_input_ids, *segments):
    input_ids = tokenizer.pad_token_id * torch.ones_like(full_input_ids)
    for c_range in segments:
        input_ids[0, c_range[0]:c_range[1]] = full_input_ids[0, c_range[0]:c_range[1]]
    prob = model.probe_forward(**base_inputs, input_ids=input_ids)
    return prob

def token_masking_feat_interaction(args, tokenizer, model, inputs, feature):
    merged_tokens, segments = merge_tokens_into_words(tokenizer, feature)
    inputs['return_kl'] = False

    full_input_ids = inputs.pop('input_ids')
    full_positioin_ids = create_position_ids_from_input_ids(full_input_ids, tokenizer.pad_token_id).to(full_input_ids.device)

    # fix position id
    inputs['position_ids'] = full_positioin_ids
    # fix cls ? maybe

    zero_input_ids = tokenizer.pad_token_id * torch.ones_like(full_input_ids)
    full_prob = model.probe_forward(**inputs, input_ids=full_input_ids)
    zero_prob = model.probe_forward(**inputs, input_ids=zero_input_ids)
    mask_t_probs = [masked_token_prob(tokenizer, model, inputs, full_input_ids, c_range) for c_range in segments]
    kept_t_probs = [kept_token_prob(tokenizer, model, inputs, full_input_ids, c_range) for c_range in segments]

    # arch_detect
    interaction_importance = []
    for i in range(len(segments)):
        range0 = segments[i]
        for j in range(i + 1, len(segments)):
            range1 = segments[j]

            # full context
            mask_t0_t1_prob = masked_token_prob(tokenizer, model, inputs, full_input_ids, range0, range1)
            full_importance = full_prob + mask_t0_t1_prob - mask_t_probs[i] - mask_t_probs[j]

            # zero context
            kept_t0_t1_prob = kept_token_prob(tokenizer, model, inputs, full_input_ids, range0, range1)
            zero_importance = kept_t0_t1_prob + zero_prob - kept_t_probs[i] - kept_t_probs[j]

            importance = (full_importance + zero_importance) / 2
            report = {'full': full_prob, 'mask_i': mask_t_probs[i], 'mask_j':  mask_t_probs[j], 'mask_i_j': mask_t0_t1_prob,
                        'zero': zero_prob, 'keep_i': kept_t_probs[i], 'keep_j': kept_t_probs[j], 'keep_i_j': kept_t0_t1_prob}
            interaction_importance.append((i, j, importance, report))    
    
    interaction_importance.sort(key=lambda x: abs(x[2]), reverse=True)
    interaction_importance = [x for x in interaction_importance]
    return interaction_importance

def predict_and_calc_interaction(args, batch, model, tokenizer, batch_features, batch_examples):
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    # only allow batch size 1
    assert batch[0].size(0) == 1    
    # run predictions
    with torch.no_grad():
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }

        if args.model_type in ["roberta", "distilbert", "camembert", "bart"]:
            del inputs["token_type_ids"]
        feature_indices = batch[3]
        outputs = model.restricted_forward(**inputs)

    batch_start_logits, batch_end_logits = outputs
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
        args.dataset
    )
    
    # run attributions
    batch_start_indexes = torch.LongTensor([x.start_index for x in batch_prelim_results]).to(args.device)
    batch_end_indexes = torch.LongTensor([x.end_index for x in batch_prelim_results]).to(args.device)
    
    # for data parallel 
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],        
        "start_indexes": batch_start_indexes,
        "end_indexes": batch_end_indexes,
        "final_start_logits": batch_start_logits,
        "final_end_logits": batch_end_logits,        
    }
    if args.model_type in ["roberta", "distilbert", "camembert", "bart"]:
        del inputs["token_type_ids"]
    
    with torch.no_grad():
        importances = token_masking_feat_interaction(args, tokenizer, model, inputs, batch_features[0])

    return batch_predictions, batch_prelim_results, importances

def arch_interp(args, model, tokenizer, prefix=""):
    if not os.path.exists(args.interp_dir):
        os.makedirs(args.interp_dir)

    # fix the model
    model.requires_grad_(False)

    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    
    # assume one on on mapping
    assert len(examples) == len(features)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = 1    
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
        batch_predictions, batch_prelim_results, batch_importances = predict_and_calc_interaction(
            args,
            batch,
            model,
            tokenizer,
            batch_features,
            batch_examples
        )
        dump_arch_info(args, batch_examples, batch_features, tokenizer, batch_predictions, batch_prelim_results, batch_importances)
        # lots of info, dump to files immediately        
        all_predictions.append(batch_predictions)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
   
    all_predictions = merge_predictions(all_predictions)
    results = hotpot_evaluate(examples[:len(all_predictions)], all_predictions)
    return results

def dump_arch_info(args, examples, features, tokenizer, predictions, prelim_results, importances):
    
    # attentions, attributions
    # N_Layer * B * N_HEAD * L * L
    # print(importances)
    for example, feature, prelim_result, importance in zip(
        examples,
        features,
        prelim_results,
        [importances]
    ):
        filename = os.path.join(args.interp_dir, f'{feature.example_index}-{feature.qas_id}.bin')
        prelim_result = prelim_result._asdict()
        prediction = predictions[example.qas_id]
        torch.save({'example': example, 'feature': feature, 'prediction': prediction, 'prelim_result': prelim_result,
            'importance':importance}, filename)

def main():
    parser = argparse.ArgumentParser()
    register_args(parser)  

    parser.add_argument("--interp_dir",default=None,type=str,required=True,help="The output directory where the model checkpoints and predictions will be written.")    

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

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
    checkpoint = args.model_name_or_path
    logger.info("Evaluate the following checkpoints: %s", checkpoint)

    # Reload the model
    model = LAtAttrRobertaForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
    model.to(args.device)

    # Evaluate
    result = arch_interp(args, model, tokenizer, prefix="")
    logger.info("Results: {}".format(result))

    return result

if __name__ == "__main__":
    main()