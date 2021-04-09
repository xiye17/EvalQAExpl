from PIL import ImageFont, ImageDraw, Image
from colour import Color
import numpy as np
import math
from itertools import chain
import os
from os.path import join
import shutil
import string

GLOBAL_W_PADDING = 600
GLOBAL_H_PADDING = 80
TEXT_GAP = 250
FONT_SIZE = 10
FONT = 'arial'
TEXT_HEIGHT = 15
TEXT_W_PADDING = 2
BG_COLOR = Color('white')
POS_COLOR = Color('blue')
NEG_COLOR = Color('red')
MERGE_THRESHOLD=0.005
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

class Element:
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class TextLink:
    def __init__(self, src_element, dst_element, val=0, x0=0, y0=0, x1=0, y1=0):
        self.src_element = src_element
        self.dst_element = dst_element
        self.val = val
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

class TextElement(Element):
    def __init__(self, text, index, x=0, y=0, w=0, h=0):
        super().__init__(x,y,w,h)
        self.text = text
        self.index = index

class LinkGraph:
    def __init__(self, tokens, connection, additional_info=None, scale=True, vis_positive=True, vis_negative=True):
        self.tokens = tokens

        sum_attr, max_attr, min_attr = np.sum(connection), np.max(connection), np.min(connection)
        # connection = np.maximum(connection, 0)
        if scale:
            connection = connection / np.max(np.abs(connection))
        
        tokens, connection = merge_unimportant_blocks(tokens, connection)

        self.connection = connection

        self.left_texts = []
        self.right_texts = []
        self.links = []

        self.left_texts = [TextElement(t, i) for (i, t) in enumerate(tokens)]
        self.right_texts = [TextElement(t, i) for (i, t) in enumerate(tokens)]

        num_tokens = len(tokens)
        for i in range(num_tokens):
            for j in range(num_tokens):
                val = connection[i, j]
                if abs(val) < PLOT_THRESHOLD: # 1/100 basically transparent
                    continue
                self.links.append(TextLink(self.left_texts[i], self.right_texts[j], val))
    
        self.title_text = 'num: {}, sum: {:.3f}, max_attr: {:.3f}, min_attr: {:.3f}'.format(len(self.links), sum_attr, max_attr, min_attr)
        self.add_info = additional_info
        self.vis_positive = vis_positive
        self.vis_negative = vis_negative

    def _arrange_texts(self, dummy_draw, font, side):
        if side == 'left':
            texts = self.left_texts
        else:
            texts = self.right_texts
        
        for i, box in enumerate(texts):
            tw, th = dummy_draw.textsize(box.text, font=font)
            box.w = tw
            box.h = th
            box.y = GLOBAL_H_PADDING + i * TEXT_HEIGHT
            if side == 'left':
                box.x = GLOBAL_W_PADDING - tw
            else:
                box.x = TEXT_GAP + GLOBAL_W_PADDING

    def _arrange_links(self, dummy_draw, font):
        for i, link in enumerate(self.links):
            src = link.src_element
            dst = link.dst_element
            link.x0, link.y0 = GLOBAL_W_PADDING + TEXT_W_PADDING,  GLOBAL_H_PADDING + (src.index + 0.5) * TEXT_HEIGHT
            link.x1, link.y1 = GLOBAL_W_PADDING + TEXT_GAP - TEXT_W_PADDING, GLOBAL_H_PADDING + (dst.index + 0.5) * TEXT_HEIGHT
            


    def get_global_size(self):
        w = GLOBAL_W_PADDING + TEXT_GAP + GLOBAL_W_PADDING
        h = GLOBAL_H_PADDING + len(self.tokens) * TEXT_HEIGHT + GLOBAL_H_PADDING
        return w, h
        

    def arrange(self, dummy_draw, font):
        self._arrange_texts(dummy_draw, font, 'left')
        self._arrange_texts(dummy_draw, font, 'right')
        self._arrange_links(dummy_draw, font)
    
    def render(self, draw, font):
        draw.text((GLOBAL_W_PADDING, 30), self.title_text, font=font, fill=(0, 0, 0))
        if self.add_info:
            draw.text((GLOBAL_W_PADDING, 30 + TEXT_HEIGHT), self.add_info, font=font, fill=(0, 0, 0))
        for b in chain(self.left_texts, self.right_texts):
            mid_y = b.y + (TEXT_HEIGHT - b.h) / 2
            draw.text((b.x, mid_y), b.text, font=font, fill=(0, 0, 0))

        for l in sorted(self.links, key=lambda x: abs(x.val)):
            if (l.val >= 0 and self.vis_positive) or (l.val < 0 and self.vis_negative):
                draw.line(((l.x0,l.y0), (l.x1, l.y1)), fill=positive_color(l.val) if l.val >= 0 else negative_color(-l.val), width=2)


def visualize_connection(filename, tokens, connection, interp_info, vis_positive=True, vis_negative=True):
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
    graph = LinkGraph(tokens, connection, additional_info=additional_info, vis_positive=vis_positive, vis_negative=vis_negative)
    global_w, global_h = graph.get_global_size()

    font = ImageFont.truetype("%s.ttf"%(FONT), FONT_SIZE)

    image = Image.new('RGB', (global_w,global_h), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    graph.arrange(draw, font)
    graph.render(draw, font)

    image.save(filename)

def merge_unimportant_blocks(tokens, connection, threshold=MERGE_THRESHOLD):
    link_as_src = np.all(np.abs(connection) < threshold, axis=0)
    link_as_dst = np.all(np.abs(connection) < threshold, axis=1)
    unimportant = np.logical_and(link_as_src, link_as_dst)
    # print('Removeable', np.sum(unimportant), unimportant.size)
    
    force_break = True
    last_flag = False
    end_points = []
    for i, f in enumerate(unimportant):
        # can be merged
        if last_flag:
            if not f:
                # do nothing
                end_points.append(i)
            last_flag = f
        else:
            end_points.append(i)
            last_flag = f
    end_points.append(unimportant.size)
    # print(end_points)

    segments = []
    for i in range(1, len(end_points)):
        if end_points[i - 1] == end_points[i]:
            continue
        segments.append((end_points[i - 1], end_points[i]))
    
    merged_tokens = []
    for s0, s1 in segments:
        merged_tokens.append(' '.join(tokens[s0:s1]))
    merged_connection = merge_attention_by_segments(connection, segments)
    return merged_tokens, merged_connection


def merge_attention_by_segments(attention, segments):
    new_val = []
    for a, b in segments:
        new_val.append(np.sum(attention[a:b, :], axis=0))
    attention = np.stack(new_val, axis=0)
    new_val = []
    for a, b in segments:
        new_val.append(np.sum(attention[:, a:b], axis=1))
    attention = np.stack(new_val, axis=1)
    return attention
