import argparse
import glob
import logging
import os
import random
import timeit
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from common.config import load_pretrained_model, load_config_and_tokenizer
from common.custom_squad_feature import custom_squad_convert_examples_to_features,  SquadResult, SquadProcessor
from common.qa_metrics import (compute_predictions_logits,hotpot_evaluate,)
from types import SimpleNamespace
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

def transform(x):
    if '#E' in x['context'] or '#P' in x['context']:
        print(x['context'])
        raise RuntimeError('Wrong Template')

    return {
      "title": x['id'],
      "paragraphs": [
        {
          "context": x['context'],
          "qas": [
            {
              "id": x['id'],
              "question": x['question'],
              "answers": [
                {
                  "answer_start": -1,
                  "text": x['answer']
                }
              ],
              "is_yesno": True,
              "question_type": "comparison"
            }
          ]
        }
      ]
    }


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def get_dummy_args(gpu='0'):
    args = SimpleNamespace()

    # Setup CUDA, GPU & distributed training
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.model_type = 'roberta-base'
    args.model_name_or_path = 'checkpoints/hpqa_roberta-base'
    args.config_name = None
    args.cache_dir = 'hf_cache'
    args.tokenizer_name = None
    args.dataset = 'hpqa'

    args.null_score_diff_threshold = 0.0
    args.n_best_size = 10
    args.max_answer_length = 30
    args.max_query_length = 60
    args.max_seq_length = 512
    args.do_lower_case = False
    args.threads = 1

    return args

class HotpotPredictor:
    def __init__(self, gpu='0'):
        self.args = get_dummy_args(gpu=gpu)
        args = self.args
        config, tokenizer = load_config_and_tokenizer(args)        

        # Reload the model
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path)
        model.to(args.device)
        model.eval

        self.config = config
        self.tokenizer = tokenizer
        self.model = model

    def predict(self, data, return_prob=False):
        args = self.args
        tokenizer = self.tokenizer
        model = self.model

        dataset, examples, features = self.load_examples(data, evaluate=True, output_examples=True)
        args.eval_batch_size =1
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        all_results = []        
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):            
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                    del inputs["token_type_ids"]

                feature_indices = batch[3]

                # XLNet and XLM use more arguments for their predictions
                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                    # for lang_id-sensitive xlm models
                    if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                        inputs.update(
                            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                        )
                outputs = model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        predictions, topk_preds = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            None,
            None,
            None,
            False,
            False,
            args.null_score_diff_threshold,
            tokenizer,
            args.dataset,
            return_nbest=True,
        )
        if return_prob:
            return predictions, topk_preds
        return predictions
    
    def load_examples(self, input_data, evaluate=False, output_examples=False):        
        processor = SquadProcessor()        
        input_data = [transform(x) for x in input_data]
        examples = processor._create_examples(input_data, 'dev', tqdm_enabled=False)

        features, dataset = custom_squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.args.max_seq_length,
            max_query_length=self.args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            tqdm_enabled=False,
            threads=self.args.threads,
            dataset=self.args.dataset,
        )

        if output_examples:
            return dataset, examples, features
        return dataset

def make_qa_data(context, question, answer, id):
    return {
        'id': id,
        'context': context,
        'question': question,
        'answer': answer
    }

def get_oringinal_prediction(meta, predictor):
    data = meta['original']
    data['id'] = 'original'
    prediction = predictor.predict([data])
    return prediction['original']


# --------------------------- utils for testers --------------------------
class HotpotTesterBase:
    split_name = 'none'
    method_name = 'none'
    def __init__(self):
        self.file_dict = self.build_file_dict()
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', cache_dir='hf_cache')        

    def build_file_dict(self):
        prefix = 'hpqa_{}-perturb_roberta-base'.format(self.split_name)
        fnames = os.listdir(join('interpretations', self.method_name, prefix))
        qa_ids = [x.split('-')[1].split('.')[0] for x in fnames]
        fullnames = [join('interpretations', self.method_name, prefix, x) for x in fnames]
        return dict(zip(qa_ids, fullnames))
        
    def load_interp(self, qas_id):
        return torch.load(self.file_dict[qas_id])
    
    def load_annotation(self, qas_id):
        meta_filename = join('hotpot_counterfactuals/{}_perturb/'.format(self.split_name), qas_id + '.json')
        meta = read_json(meta_filename)
        return meta

def list_whole_word_match(l, k, start):
    for p in range(len(k)):
        if l[start + p] != k[p]:
            return False

    # print(l[start + len(k)], l[start + len(k)][0], l[start + len(k)][0].isalnum())
    if (start + len(k)) >= len(l):
        return True
    leading_char =l[start + len(k)][0]
    if leading_char != 'Ġ' and leading_char.isalnum():
        return False    
    return True

    
def extract_token_segments(tokenizer, interp, tok, include_question=True, include_context=True):
    sub_tokens = tokenizer.tokenize(tok, add_prefix_space=True)
    feature = interp['feature']
    doc_tokens = feature.tokens


    context_start = doc_tokens.index(tokenizer.eos_token)
    range_left = 0 if include_question else context_start
    range_right = len(doc_tokens) if include_context else context_start

    start_positions = [i for i in range(range_left, range_right) if list_whole_word_match(doc_tokens, sub_tokens, i)]

    segments = [(s, s + len(sub_tokens)) for s in start_positions]
    
    # special case
    sub_tokens = tokenizer.tokenize(tok)
    if list_whole_word_match(doc_tokens, sub_tokens, 1):
        segments = [(1, 1 + len(sub_tokens))] + segments
    return segments

def aggregate_token_attribution_from_link(interp):
    attribution = interp['attribution']
    attribution_val = attribution.numpy()
    # attribution_val[attribution_val < 0] = 0
    aggregated_attribution = np.sum(attribution_val, axis=0)
    # aggregated_attribution = np.max(attribution_val, axis=0)
    aggregated_attribution = np.sum(aggregated_attribution, axis=0)
    aggregated_attribution[aggregated_attribution < 0] = 0
    # aggregated_attribution = np.abs(aggregated_attribution)

    diag_attribution = np.diag(aggregated_attribution)
    gather_weight = np.sum(aggregated_attribution, axis=1)
    dispatch_weight = np.sum(aggregated_attribution, axis=0)
    agg_weight = (gather_weight + dispatch_weight)
    
    return agg_weight
