 
import os
import torch
from os.path import join
from transformers import AutoTokenizer
import re
import itertools
import numpy as np

from common.utils import read_json
from common.dataset_utils import merge_tokens_into_words
from eval_hotpot_exp.utils import (
    HotpotTesterBase,
    extract_token_segments,
    aggregate_token_attribution_from_link,
    extract_segments_from_merged_tokens,
    aggregate_token_attribution_from_arch
)

def aggregated_link_attribution_in_question(tokenizer, interp, targets):
    target_segments = []
    for tok in targets:
        target_segments.extend(extract_token_segments(tokenizer, interp, tok, include_context=False))
    if len(target_segments) == 0:
        print( tokenizer.tokenize(targets[0], add_prefix_space=True))
        print(interp['feature'].tokens)
        print(interp['feature'].qas_id)
        exit()
    # print(targets)
    # for a, b in target_segments:
    #     print(interp['feature'].tokens[a:b])
    doc_tokens = interp['feature'].tokens
    context_start = doc_tokens.index(tokenizer.eos_token)
    if doc_tokens[context_start - 1] in ['?', 'Ġ?']:
        context_start = context_start - 1
    token_attribution = aggregate_token_attribution_from_link(interp)

    selected_idx = list(itertools.chain(*[list(range(a,b)) for (a,b) in target_segments]))
    selected_attribution = token_attribution[selected_idx]


    return_val = np.sum(selected_attribution) / np.sum(token_attribution[1:context_start])
    return return_val

def aggregated_token_attribution_in_question(tokenizer, interp, targets):
    target_segments = []
    for tok in targets:
        target_segments.extend(extract_token_segments(tokenizer, interp, tok, include_context=False))
    if len(target_segments) == 0:
        print( tokenizer.tokenize(targets[0], add_prefix_space=True))
        print(interp['feature'].tokens)
        print(interp['feature'].qas_id)
        raise RuntimeError('Error in matching')
    # print(targets)
    # for a, b in target_segments:
    #     print(interp['feature'].tokens[a:b])
    doc_tokens = interp['feature'].tokens
    context_start = doc_tokens.index(tokenizer.eos_token)
    if doc_tokens[context_start - 1] in ['?', 'Ġ?']:
        context_start = context_start - 1
    token_attribution = interp['attribution'].numpy()
    token_attribution[token_attribution < 0] = 0
    selected_idx = list(itertools.chain(*[list(range(a,b)) for (a,b) in target_segments]))
    selected_attribution = token_attribution[selected_idx]

    return_val = np.sum(selected_attribution) / np.sum(token_attribution[1:context_start])
    return return_val

def aggregated_token_arch_importance_in_question(tokenizer, interp, targets):
    merged_tokens, merged_segments = merge_tokens_into_words(tokenizer, interp['feature'])
    merged_tokens = [x.lstrip() for x in merged_tokens]
    target_segments = []
    for tok in targets:
        target_segments.extend(extract_segments_from_merged_tokens(tokenizer, tok, merged_tokens, include_context=False))


    doc_tokens = interp['feature'].tokens
    context_start = doc_tokens.index(tokenizer.eos_token)

    if doc_tokens[context_start - 1] in ['?', 'Ġ?']:
        context_start = context_start - 1

    token_attribution = aggregate_token_attribution_from_arch(interp, merged_tokens)
    context_start = merged_tokens.index(tokenizer.eos_token)
    selected_idx = list(itertools.chain(*[list(range(a,b)) for (a,b) in target_segments]))    
    selected_attribution = token_attribution[selected_idx]

    return_val = np.sum(selected_attribution) / np.sum(token_attribution[(context_start + 1):])
    return return_val

class BridgeTesterBase(HotpotTesterBase):
    split_name = 'bridge'
    def get_impacts_of_primary_question(self, interp_info, annotation):
        raise NotImplementedError

class BridgeConfTester(BridgeTesterBase):
    method_name = 'conf'
    def __init__(self):
        self.file_dict = self.build_file_dict()

    def build_file_dict(self):
        prefix = 'hpqa_bridge_mannual_predictions.json'
        fname = join('interpretations', prefix)
        return read_json(fname)
        
    def load_interp(self, qas_id):
        return self.file_dict[qas_id]

    def get_impacts_of_primary_question(self, interp_info, annotation):
        return -interp_info[0]['probability']

class BridgeAtAttrTester(BridgeTesterBase):
    method_name = 'atattr'
    def __init__(self):
        super().__init__()    

    def get_impacts_of_primary_question(self, interp_info, annotation):
        primary_question = [annotation['original']['primary_question']]
        return aggregated_link_attribution_in_question(self.tokenizer, interp_info, primary_question)


class BridgeLAtAttrTester(BridgeTesterBase):
    method_name = 'latattr'
    def __init__(self):
        super().__init__() 
    
    def get_impacts_of_primary_question(self, interp_info, annotation):
        primary_question = [annotation['original']['primary_question']]
        return aggregated_link_attribution_in_question(self.tokenizer, interp_info, primary_question)

class BridgeTokIGTester(BridgeTesterBase):
    method_name = 'tokig'
    def __init__(self):
        super().__init__() 

    def get_impacts_of_primary_question(self, interp_info, annotation):
        primary_question = [annotation['original']['primary_question']]
        return aggregated_token_attribution_in_question(self.tokenizer, interp_info, primary_question)

class BridgeShapTester(BridgeTokIGTester):
    method_name = 'shap'

class BridgeLimeTester(BridgeTokIGTester):
    method_name = 'lime'

class BridgeArchTester(BridgeTesterBase):
    method_name = 'arch'
    def __init__(self):
        super().__init__() 

    def get_impacts_of_primary_question(self, interp_info, annotation):
        primary_question = [annotation['original']['primary_question']]
        return aggregated_token_arch_importance_in_question(self.tokenizer, interp_info, primary_question)