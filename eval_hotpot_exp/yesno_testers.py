
import os
import torch
from os.path import join
from transformers import AutoTokenizer
import re
import itertools
import numpy as np

from common.utils import read_json
from common.dataset_utils import merge_tokens_into_words
from eval_hotpot_exp.utils import HotpotTesterBase, extract_token_segments, aggregate_token_attribution_from_link

def aggregated_link_attribution_in_context(tokenizer, interp, targets):
    target_segments = []
    for tok in targets:
        target_segments.extend(extract_token_segments(tokenizer, interp, tok, include_question=False))
    # print(targets)
    # for a, b in target_segments:
    #     print(interp['feature'].tokens[a:b])

    token_attribution = aggregate_token_attribution_from_link(interp)
    doc_tokens = interp['feature'].tokens
    context_start = doc_tokens.index(tokenizer.eos_token)
    selected_idx = list(itertools.chain(*[list(range(a,b)) for (a,b) in target_segments]))
    selected_attribution = token_attribution[selected_idx]

    return_val = np.sum(selected_attribution) / np.sum(token_attribution[(context_start + 1):])
    return return_val


def aggregated_token_attribution_in_context(tokenizer, interp, targets):
    target_segments = []
    for tok in targets:
        target_segments.extend(extract_token_segments(tokenizer, interp, tok, include_question=False))
    # print(targets)
    # for a, b in target_segments:
    #     print(interp['feature'].tokens[a:b])

    token_attribution = interp['attribution'].numpy()
    token_attribution[token_attribution < 0] = 0
    doc_tokens = interp['feature'].tokens
    context_start = doc_tokens.index(tokenizer.eos_token)

    selected_idx = list(itertools.chain(*[list(range(a,b)) for (a,b) in target_segments]))    
    selected_attribution = token_attribution[selected_idx]

    return_val = np.sum(selected_attribution) / np.sum(token_attribution[(context_start + 1):])
    return return_val

def extract_segments_from_merged_tokens(tokenizer, target, merged_tokens, include_question=True, include_context=True):
    sub_tokens = target.split()
    context_start = merged_tokens.index(tokenizer.eos_token)
    range_left = 0 if include_question else context_start
    range_right = len(merged_tokens) if include_context else context_start
    len_sub_tokens = len(sub_tokens)
    # match_func = lambda x: ' '.join(merged_tokens[x: (x + len_sub_tokens)]) == target
    def match_func(x):
        # print(target, '--Match--', '[' +' '.join(merged_tokens[x: (x + len_sub_tokens)]) + ']')
        return (' '.join(merged_tokens[x: (x + len_sub_tokens)]) == target)
    start_positions = [i for i in range(range_left, range_right) if match_func(i)]
    segments = [(s, s + len(sub_tokens)) for s in start_positions]
    return segments

def aggregate_token_attribution_from_arch(interp, merged_tokens):
    importance = interp['importance']

    attribution = np.zeros((len(merged_tokens), len(merged_tokens)))
    for (i, j, imp, report) in importance:
        attribution[i, j] = report['keep_i_j'] - report['zero']
    aggregated_attribution = attribution
    aggregated_attribution[aggregated_attribution < 0] = 0
    gather_weight = np.sum(aggregated_attribution, axis=1)
    dispatch_weight = np.sum(aggregated_attribution, axis=0)
    agg_weight = (gather_weight + dispatch_weight)    
    return agg_weight

def aggregated_token_arch_importance_in_context(tokenizer, interp, targets):
    merged_tokens, merged_segments = merge_tokens_into_words(tokenizer, interp['feature'])
    merged_tokens = [x.lstrip() for x in merged_tokens]
    target_segments = []
    for tok in targets:
        target_segments.extend(extract_segments_from_merged_tokens(tokenizer, tok, merged_tokens, include_question=False))
    # print(targets)
    # print(target_segments)
    # for a, b in target_segments:
    #     print(interp['feature'].tokens[a:b])

    token_attribution = aggregate_token_attribution_from_arch(interp, merged_tokens)
    context_start = merged_tokens.index(tokenizer.eos_token)
    selected_idx = list(itertools.chain(*[list(range(a,b)) for (a,b) in target_segments]))    
    selected_attribution = token_attribution[selected_idx]

    return_val = np.sum(selected_attribution) / np.sum(token_attribution[(context_start + 1):])
    return return_val

class YesNoTesterBase(HotpotTesterBase):
    split_name = 'yesno'

    def get_impacts_of_property(self, interp_info, annotation):
        raise NotImplementedError

class YesNoAtAttrTester(YesNoTesterBase):
    method_name = 'atattr'
    def __init__(self):
        super().__init__()

    def get_impacts_of_property(self, interp_info, annotation):
        properties = annotation['perturb_property']['original_properties']
        properties = list(set(properties))
        return aggregated_link_attribution_in_context(self.tokenizer, interp_info, properties)

class YesNoLAtAttrTester(YesNoTesterBase):
    method_name = 'latattr'
    def __init__(self):
        super().__init__() 

    def get_impacts_of_property(self, interp_info, annotation):
        properties = annotation['perturb_property']['original_properties']
        properties = list(set(properties))
        return aggregated_link_attribution_in_context(self.tokenizer, interp_info, properties)
        

class YesNoTokIGTester(YesNoTesterBase):
    method_name = 'tokig'
    def __init__(self):
        super().__init__() 

    def get_impacts_of_property(self, interp_info, annotation):
        properties = annotation['perturb_property']['original_properties']
        properties = list(set(properties))
        return aggregated_token_attribution_in_context(self.tokenizer, interp_info, properties)

class YesNoArchTester(YesNoTesterBase):
    method_name = 'arch'
    def __init__(self):
        super().__init__()

    def get_impacts_of_property(self, interp_info, annotation):
        properties = annotation['perturb_property']['original_properties']
        properties = list(set(properties))
        return aggregated_token_arch_importance_in_context(self.tokenizer, interp_info, properties)
