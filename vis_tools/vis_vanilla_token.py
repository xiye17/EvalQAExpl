from PIL import ImageFont, ImageDraw, Image
from colour import Color
import numpy as np
import math
from itertools import chain
import os
from os.path import join
import shutil
import string
from .vis_token import WeightedTextElement, positive_color, negative_color

GLOBAL_W_PADDING = 200
GLOBAL_H_PADDING = 80
CONTEXT_W = 800
FONT_SIZE = 10
FONT = 'arial'
TEXT_HEIGHT = 15
TEXT_W_PADDING = 4

class VanillaTokenGraph:
    def __init__(self, tokens, attributions, additional_info=None):
        self.tokens = tokens

        sum_attr, max_attr, min_attr = np.sum(attributions), np.max(attributions), np.min(attributions)
        max_connection_abs_val = np.max(np.abs(attributions))
        # self.connection = connection
        # aggregated
        agg_weight = attributions
        # draw figure
        self.aggregated_boxes = []
        max_agg_val = np.max(np.abs(agg_weight))
        for i, (tok, w) in enumerate(zip(tokens, agg_weight)):
            box = WeightedTextElement(tok, w, i, w/max_agg_val)
            self.aggregated_boxes.append(box)

        self.title_text =  'sum: {:.3f}, max_attr: {:.3f}, min_attr: {:.3f}'.format(sum_attr, max_attr, min_attr)
        self.add_info = additional_info
    
        self.w = 0
        self.h = 0

    def get_global_size(self):
        w = GLOBAL_W_PADDING + CONTEXT_W + GLOBAL_W_PADDING
        # estimation
        h = GLOBAL_H_PADDING + 300 + GLOBAL_H_PADDING
        return w, h
        
    def _arrange(self, boxes, draw, font, start_x, start_y):
        line_h = TEXT_HEIGHT * 2 
        line = 0
        cursor = start_x
        for i, box in enumerate(boxes):
            bw, bh = box.box_size(draw, font)

            # new line
            if cursor + bw > start_x + CONTEXT_W:
                line += 1
                cursor = start_x
                bx = cursor
            else:
                bx = cursor
            cursor += bw + TEXT_W_PADDING
            by = line * TEXT_HEIGHT * 2 + start_y
            box.set_position(bx, by)

        
        ending_h = start_y + (line + 1) * line_h
        return ending_h

    def arrange(self, draw, font):
        endding = GLOBAL_H_PADDING
        endding = self._arrange(self.aggregated_boxes, draw, font, GLOBAL_W_PADDING, endding)

    def render(self, draw, font):
        draw.text((GLOBAL_W_PADDING, 30), self.title_text, font=font, fill=(0, 0, 0))
        if self.add_info:
            draw.text((GLOBAL_W_PADDING, 30 + TEXT_HEIGHT), self.add_info, font=font, fill=(0, 0, 0))
        
        for box in self.aggregated_boxes:
            box.render(draw, font)


def visualize_vanilla_tok_attribution(filename, tokens, attributions, interp_info):
    # connection should be n_toks * n_toks
    assert len(tokens) == attributions.size
    
    if interp_info['example'].answer_text:
        gt_text = interp_info['example'].answer_text
    elif len(interp_info['example'].answers):
        gt_text = interp_info['example'].answers[0]['text']
    else:
        gt_text = "NULL"

    p_text = interp_info['prediction']
    additional_info = 'P: {}        G: {}'.format(p_text, gt_text)    
    graph = VanillaTokenGraph(tokens, attributions, additional_info=additional_info)
    global_w, global_h = graph.get_global_size()

    font = ImageFont.truetype("%s.ttf"%(FONT), FONT_SIZE)

    image = Image.new('RGB', (global_w,global_h), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    graph.arrange(draw, font)
    graph.render(draw, font)

    image.save(filename)
