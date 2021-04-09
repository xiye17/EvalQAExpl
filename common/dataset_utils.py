import os
import string
from os.path import join
from collections import OrderedDict

from common.utils import read_json
from transformers import RobertaTokenizer

def get_prefix_tokens(dataset, tokenizer):
    if dataset == 'hpqa':
        return ['yes', 'no', 'unk', tokenizer.sep_token]
    elif dataset == 'squad':
        return []
    elif dataset == 'synth':
        return []    
    else:
        raise RuntimeError('invalid dataset')

def _merge_roberta_tokens_into_words(tokenizer, feature):
    tokens = feature.tokens

    decoded_each_tok = [
        bytearray([tokenizer.byte_decoder[c] for c in t]).decode("utf-8", errors=tokenizer.errors) for t in tokens
    ]

    token_to_orig_map = feature.token_to_orig_map

    end_points = []
    context_start = tokens.index(tokenizer.eos_token)
    force_break = False
    for i, t in enumerate(decoded_each_tok):
        # special token
        if t in tokenizer.all_special_tokens:
            end_points.append(i)
            force_break = True
            continue

        if t in string.punctuation:
            end_points.append(i)
            force_break = True
            continue

        if force_break:
            end_points.append(i)
            force_break = False
            continue

        # if in question segment
        if i <= context_start:
            if t[0] == ' ':
                decoded_each_tok[i] = t[1:]
                end_points.append(i)
        else:
            if token_to_orig_map[i] != token_to_orig_map[i - 1]:
                end_points.append(i)
    end_points.append(len(decoded_each_tok))

    # if in context segment
    segments = []
    for i in range(1, len(end_points)):
        if end_points[i - 1] == end_points[i]:
            continue
        segments.append((end_points[i - 1], end_points[i]))
    
    merged_tokens = []
    for s0, s1 in segments:
        merged_tokens.append(''.join(decoded_each_tok[s0:s1]))
    
    return merged_tokens, segments

def _merge_simple_tokens_into_words(tokenizer, feature):
    tokens = feature.tokens
    segments = [(i,i+1) for i in range(len(tokens))]
    return tokens, segments

def merge_tokens_into_words(tokenizer, feature):
    if isinstance(tokenizer, RobertaTokenizer):
        return _merge_roberta_tokens_into_words(tokenizer, feature)
    else:
        return _merge_simple_tokens_into_words(tokenizer, feature)

def read_hotpot_perturbations(split):
    assert split in ['bridge', 'yesno']
    prefix = f'hotpot_counterfactuals/{split}'
    fnames = os.listdir(prefix)
    fnames.sort()    

    annotation_dict = OrderedDict()
    for i,fname in enumerate(fnames):
        meta = read_json(join(prefix, fname))
        meta['quick_id'] = str(i)
        annotation_dict[meta['id']] = meta
    return annotation_dict
