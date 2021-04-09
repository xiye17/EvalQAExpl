from PIL import ImageFont, ImageDraw, Image
from colour import Color
import numpy as np
import math
from itertools import chain
import os
from os.path import join
import shutil
import string
from transformers import RobertaTokenizer

from .vis_attention import visualize_connection, merge_attention_by_segments
from .vis_token import visualize_tok_attribution, merge_token_attribution_by_segments
from .vis_vanilla_token import visualize_vanilla_tok_attribution
from common.dataset_utils import merge_tokens_into_words

def visualize_token_attributions(args, tokenizer, interp_info):
    assert args.visual_dir is not None
    feature = interp_info['feature']

    # prefix    
    prefix = join(args.visual_dir, f'{feature.example_index}-{feature.qas_id}')

    # attribution
    # N Layer * N Head
    attribution = interp_info['attribution']
    prelim_result = interp_info['prelim_result']
    # attribution = attribution.res

    attribution_val = attribution.numpy()
    n_tokens = attribution_val.size

    words, segments = merge_tokens_into_words(tokenizer, interp_info['feature'])
    
    # plot aggregated
    # along layers
    aggregated_attribution = merge_token_attribution_by_segments(attribution_val, segments)
    visualize_vanilla_tok_attribution(prefix + '.jpg', words, aggregated_attribution, interp_info)

def visualize_attention_attributions(args, tokenizer, interp_info, do_head=False, do_layer=False):
    assert args.visual_dir is not None
    feature = interp_info['feature']

    # prefix    
    prefix = join(args.visual_dir, f'{feature.example_index}-{feature.qas_id}')
    # _mkdir_f(prefix)

    # attribution
    # N Layer * N Head
    attribution = interp_info['attribution']
    attention = interp_info['attention']
    prelim_result = interp_info['prelim_result']
    # attribution = attribution.res
    n_layers, n_heads, n_tokens, _ = tuple(attribution.size())

    attribution_val = attribution.numpy()
    attribution_diff = np.sum(attribution_val)
    attribution_val = attribution_val / attribution_diff

    words, segments = merge_tokens_into_words(tokenizer, interp_info['feature'])
    
    # plot aggregated
    # along layers
    aggregated_attribution = np.sum(attribution_val, axis=0)
    aggregated_attribution = np.sum(aggregated_attribution, axis=0)
    aggregated_attribution = merge_attention_by_segments(aggregated_attribution, segments)    
    visualize_connection(prefix + '-attn_aggregated.jpg', words, aggregated_attribution, interp_info)
    visualize_tok_attribution(prefix + '-token_attribution.jpg', words, aggregated_attribution, interp_info)
    if do_head:
        aggregated_by_head = np.sum(attribution_val, axis=0)
        for i_head in range(aggregated_by_head.shape[0]):
            aggregated_head_i = aggregated_by_head[i_head]
            aggregated_head_i = merge_attention_by_segments(aggregated_head_i, segments)
            visualize_connection(prefix + f'-xhead-{i_head}.jpg', words, aggregated_head_i, interp_info)
    if do_layer:    
        aggregated_by_layer = np.sum(attribution_val, axis=1)
        for i_layer in range(aggregated_by_layer.shape[0]):
            aggregated_layer_i = aggregated_by_layer[i_layer]
            aggregated_layer_i = merge_attention_by_segments(aggregated_layer_i, segments)
            visualize_connection(prefix + f'-xlayer-{i_layer}.jpg', words, aggregated_layer_i, interp_info)

def visualize_pruned_layer_attributions(args, tokenizer, interp_info, do_layer=True):
    assert args.visual_dir is not None
    feature = interp_info['feature']

    # prefix    
    prefix = join(args.visual_dir, f'{feature.example_index}-{feature.qas_id}')
    _mkdir_f(prefix)

    # attribution
    # N Layer * N Head
    attribution = interp_info['attribution']
    attention = interp_info['attention']
    prelim_result = interp_info['prelim_result']
    # attribution = attribution.res
    n_layers, n_heads, n_tokens, _ = tuple(attribution.size())

    attribution_val = attribution.numpy()
    
    active_layers = interp_info['active_layers']
    words, segments = merge_tokens_into_words(tokenizer, interp_info['feature'])
    if do_layer:    
        aggregated_by_layer = np.sum(attribution_val, axis=1)
        for i_layer in range(aggregated_by_layer.shape[0]):
            if not active_layers[i_layer]:
                continue
            aggregated_layer_i = aggregated_by_layer[i_layer]
            aggregated_layer_i = merge_attention_by_segments(aggregated_layer_i, segments)
            visualize_connection(join(prefix, f'layer-{i_layer}.jpg'), words, aggregated_layer_i, interp_info, vis_negative=False)
