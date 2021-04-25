
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

def aggregated_token_arch_importance_in_context(tokenizer, interp, targets):
    merged_tokens, merged_segments = merge_tokens_into_words(tokenizer, interp['feature'])
    merged_tokens = [x.lstrip() for x in merged_tokens]
    target_segments = []
    for tok in targets:
        target_segments.extend(extract_segments_from_merged_tokens(tokenizer, tok, merged_tokens, include_question=False))

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

class YesNoShapTester(YesNoTokIGTester):
    method_name = 'shap'

class YesNoLimeTester(YesNoTokIGTester):
    method_name = 'lime'

class YesNoArchTester(YesNoTesterBase):
    method_name = 'arch'
    def __init__(self):
        super().__init__()

    def get_impacts_of_property(self, interp_info, annotation):
        properties = annotation['perturb_property']['original_properties']
        properties = list(set(properties))
        return aggregated_token_arch_importance_in_context(self.tokenizer, interp_info, properties)
