from PIL import ImageFont, ImageDraw, Image
from colour import Color
import numpy as np
import math
from itertools import chain
import os
from os.path import join
import shutil
import string
from .vis_attention import Element, TextElement, merge_unimportant_blocks

GLOBAL_W_PADDING = 200
GLOBAL_H_PADDING = 80
CONTEXT_W = 800
FONT_SIZE = 10
FONT = 'arial'
TEXT_HEIGHT = 15
TEXT_W_PADDING = 4
BG_COLOR = Color('white')
POS_COLOR = Color('blue')
NEG_COLOR = Color('red')
MERGE_THRESHOLD=0.001
PLOT_THRESHOLD=0.001


def positive_color(val):
    val = min(val, 1.0)
    val = max(val, 0.0)
    new_c = []
    for a, b in zip(POS_COLOR.rgb, BG_COLOR.rgb):
        c = val * a + (1 - val) * b
        new_c.append(c)
    return tuple(int(x * 255) for x in new_c)

def negative_color(val):
    val = min(val, 1.0)
    val = max(val, 0.0)
    new_c = []
    for a, b in zip(NEG_COLOR.rgb, BG_COLOR.rgb):
        c = val * a + (1 - val) * b
        new_c.append(c)
    return tuple(int(x * 255) for x in new_c)

class WeightedTextElement(Element):
    def __init__(self, token, weight, index, intense, x=0, y=0, w=0, h=0):
        super().__init__(x,y,w,h)
        self.token = token
        weight_str = '%.3f' % weight
        if weight >= 0:
            self.weight = weight_str[1:]
        else:
            self.weight = weight_str[2:]
        self.index = index
        self.token_box = TextElement(self.token, index)
        self.weight_box = TextElement(self.weight, index)
        self.intense = intense
    
    def box_size(self, draw, font):
        tw, th = draw.textsize(self.token, font=font)
        ww, wh = draw.textsize(self.weight, font=font)
        self.token_box.w = tw
        self.weight_box.w = ww
        self.w = max(tw, ww)
        self.h = 2 * TEXT_HEIGHT
        return self.w, self.h

    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.token_box.x = x + (self.w - self.token_box.w) / 2
        self.token_box.y = y
        self.weight_box.x = x + (self.w - self.weight_box.w) / 2
        self.weight_box.y = y + TEXT_HEIGHT
    
    def render(self, draw, font):
        if self.intense > 0:
            intense = self.intense
            rescaled_intense = 1.0 * intense + 0.2 * (1 - intense)
            c = positive_color(rescaled_intense)
        else:
            intense = -self.intense
            rescaled_intense = 1.0 * intense + 0.2 * (1 - intense)
            c = negative_color(rescaled_intense)
        # black = (0,0,0)
        # draw.text((self.token_box.x, self.token_box.y), self.token_box.text, font=font, fill=(0, 0, 0))
        draw.text((self.token_box.x, self.token_box.y), self.token_box.text, font=font, fill=c)
        draw.text((self.weight_box.x, self.weight_box.y), self.weight_box.text, font=font, fill=c)


class TokenGraph:
    def __init__(self, tokens, connection, additional_info=None):
        self.tokens = tokens

        sum_attr, max_attr, min_attr = np.sum(connection), np.max(connection), np.min(connection)
        max_connection_abs_val = np.max(np.abs(connection))
        tokens, connection = merge_unimportant_blocks(tokens, connection, threshold=(max_connection_abs_val * MERGE_THRESHOLD))
        self.connection = connection
        # aggregated
        gather_weight = np.sum(connection, axis=1)
        dispatch_weight = np.sum(connection, axis=0)
        agg_weight = (gather_weight + dispatch_weight) / 2
        # draw figure
        self.aggregated_boxes = []
        max_agg_val = np.max(np.abs(agg_weight))
        for i, (tok, w) in enumerate(zip(tokens, agg_weight)):
            box = WeightedTextElement(tok, w, i, w/max_agg_val)
            self.aggregated_boxes.append(box)

        self.gather_boxes = []
        max_gather_val = np.max(np.abs(gather_weight))
        for i, (tok, w) in enumerate(zip(tokens, gather_weight)):
            box = WeightedTextElement(tok, w, i, w/max_gather_val)
            self.gather_boxes.append(box)
        
        self.dispatch_boxes = []
        max_dispatch_val = np.max(np.abs(dispatch_weight))
        for i, (tok, w) in enumerate(zip(tokens, dispatch_weight)):
            box = WeightedTextElement(tok, w, i, w/max_dispatch_val)
            self.dispatch_boxes.append(box)

        self.title_text = 'Aggregated   -   Gather (X attends to)   -   Dispatach (X is attended)'
        self.add_info = additional_info
    
        self.w = 0
        self.h = 0

    def get_global_size(self):
        w = GLOBAL_W_PADDING + CONTEXT_W + GLOBAL_W_PADDING
        # estimation
        h = GLOBAL_H_PADDING + 1500 + GLOBAL_H_PADDING
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
        endding += GLOBAL_H_PADDING
        endding = self._arrange(self.gather_boxes, draw, font, GLOBAL_W_PADDING, endding)
        endding += GLOBAL_H_PADDING
        endding = self._arrange(self.dispatch_boxes, draw, font, GLOBAL_W_PADDING, endding)

    def render(self, draw, font):
        draw.text((GLOBAL_W_PADDING, 30), self.title_text, font=font, fill=(0, 0, 0))
        if self.add_info:
            draw.text((GLOBAL_W_PADDING, 30 + TEXT_HEIGHT), self.add_info, font=font, fill=(0, 0, 0))
        
        for box in self.aggregated_boxes:
            box.render(draw, font)
        
        for box in self.gather_boxes:
            box.render(draw, font)
        
        for box in self.dispatch_boxes:
            box.render(draw, font)

def visualize_tok_attribution(filename, tokens, connection, interp_info):
    # connection should be n_toks * n_toks
    assert (len(tokens),len(tokens)) == connection.shape
    
    if interp_info['example'].answer_text:
        gt_text = interp_info['example'].answer_text
    elif len(interp_info['example'].answers):
        gt_text = interp_info['example'].answers[0]['text']
    else:
        gt_text = "NULL"

    p_text = interp_info['prediction']
    additional_info = 'P: {}        G: {}'.format(p_text, gt_text)    
    graph = TokenGraph(tokens, connection, additional_info=additional_info)
    global_w, global_h = graph.get_global_size()

    font = ImageFont.truetype("%s.ttf"%(FONT), FONT_SIZE)

    image = Image.new('RGB', (global_w,global_h), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    graph.arrange(draw, font)
    graph.render(draw, font)

    image.save(filename)


def merge_token_attribution_by_segments(attributions, segments):
    new_val = []
    for a, b in segments:
        new_val.append(np.sum(attributions[a:b]))
    attention = np.array(new_val)
    return attention
