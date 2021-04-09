    
import argparse
from transformers import AutoTokenizer
from common.utils import *
from common.dataset_utils import read_hotpot_perturbations
from os.path import join
from tqdm import tqdm
import sys
# from dataset import *
import itertools
import re

def add_preproc_args(parser):
    parser.add_argument('--do_mini', action='store_true', default=False, help='Test with mini dataset for debugging')    
    parser.add_argument('--model_name', type=str, default='roberta-large', help='Model choosen')
    parser.add_argument('--cache_dir', default='hf_cache', type=str, help='custom cache dir')
    return parser

def _parse_args():
    parser = argparse.ArgumentParser()
    add_preproc_args(parser)
    args = parser.parse_args()
    return args

def construct_context_and_title(documents, supporting_titles, answer_text):
    is_span_type = (answer_text not in ['yes', 'no'])
        
    supporting_docs = [x for x in documents if x[0] in supporting_titles]
    supporting_docs = [(x[0], ''.join(x[1])) for x in supporting_docs]
    if is_span_type:
        if not any([answer_text in x[1] for x in supporting_docs]):
            raise RuntimeError('Cannot find ans')

    # if is_span_type:
    #     supporting_docs.sort(key=lambda x: answer_text in x[1], reverse=True)
    
    title = f'{supporting_docs[0][0]}, {supporting_docs[1][0]}'
    context = f'{supporting_docs[0][1]} {supporting_docs[1][1]}'
    return context, title

def construct_context_and_title(documents, supporting_titles, answer_text):
    is_span_type = (answer_text not in ['yes', 'no'])

    supporting_docs = [x for x in documents if x[0] in supporting_titles]
    supporting_docs = [(x[0], ''.join(x[1])) for x in supporting_docs]
    if is_span_type:
        if not any([answer_text in x[1] for x in supporting_docs]):
            raise RuntimeError('Cannot find ans')

    # if is_span_type:
    #     supporting_docs.sort(key=lambda x: answer_text in x[1], reverse=True)

    title = f'{supporting_docs[0][0]}, {supporting_docs[1][0]}'
    context = f'{supporting_docs[0][1]} {supporting_docs[1][1]}'
    return context, title

def construct_context_and_title_with_prediction(documents, pred):
    assert len(pred) == 2
    assert pred[0] < pred[1]
    supporting_docs = [documents[pred[0]], documents[pred[1]]]
    supporting_docs = [(x[0], ''.join(x[1])) for x in supporting_docs]
    title = f'{supporting_docs[0][0]}, {supporting_docs[1][0]}'
    context = f'{supporting_docs[0][1]} {supporting_docs[1][1]}'
    return context, title

def construct_answers(context, answer):
    if answer in ['yes', 'no']:
        return [{'answer_start': -1, 'text': answer}]
    else:
        start_positions = [i for i in range(len(context)) if context[i:].startswith(answer)]
        return [{'answer_start': i, 'text': answer} for i in start_positions]
    
def preprocess_train_example(raw_data, tokenzier, args, ranking_dict=None):
    # we'll a have a single paragraph
    answer_text = raw_data['answer']

    if ranking_dict is None:
        supporting_facts = raw_data['supporting_facts']
        supporting_titles = [x[0] for x in supporting_facts]
        context, title = construct_context_and_title(raw_data['context'], supporting_titles, answer_text)
    else:
        context, title = construct_context_and_title_with_prediction(raw_data['context'], ranking_dict[raw_data['_id']])

    if context is None:
        raise RuntimeError('Empty context')

    qa0 = {}
    qa0['id'] = raw_data['_id']
    qa0['question'] = raw_data['question']
    # sanity check
    # print(supporting_titles)
    answers = construct_answers(context, answer_text)
    if answers is None or len(answers) == 0:
        raise RuntimeError('Inviad ans')
    qa0['answers'] = answers
    qa0['is_yesno'] = answer_text in ['yes', 'no']
    qa0['question_type'] = raw_data['type']
    pargraph0 = {'context': context, 'qas': [qa0]}
    data = {'title': title, 'paragraphs': [pargraph0]}

    return data

def preprocess_eval_example(raw_data, tokenzier, args, ranking_dict=None):
    # we'll a have a single paragraph
    answer_text = raw_data['answer']

    if ranking_dict is None:
        supporting_facts = raw_data['supporting_facts']
        supporting_titles = [x[0] for x in supporting_facts]
        context, title = construct_context_and_title(raw_data['context'], supporting_titles, answer_text)
    else:
        context, title = construct_context_and_title_with_prediction(raw_data['context'], ranking_dict[raw_data['_id']])

    if context is None:
        raise RuntimeError('Empty context')

    qa0 = {}
    qa0['id'] = raw_data['_id']
    qa0['question'] = raw_data['question']
    # sanity check
    # print(supporting_titles)
    answers = [{'answer_start': -1, 'text': answer_text}]
    qa0['answers'] = answers
    qa0['is_yesno'] = answer_text in ['yes', 'no']
    qa0['question_type'] = raw_data['type']
    pargraph0 = {'context': context, 'qas': [qa0]}
    data = {'title': title, 'paragraphs': [pargraph0]}

    return data

# top  level [data, version]
def preprocess_split(raw_fname, tokenzier, args, is_train=False, ranking_dict=None):
    # read json
    raw_dataset = read_json(raw_fname)
    if args.do_mini:
        raw_dataset = raw_dataset[:32]

    data = {}
    data['version'] = '1.1'
    # data['data'] = 

    proc_dataset = []
    for raw_data in tqdm(raw_dataset, desc='Preprocessing', file=sys.stdout, total=len(raw_dataset)):
        if is_train:
            proc_data = preprocess_train_example(raw_data, tokenzier, args, ranking_dict=ranking_dict)
        else:
            proc_data = preprocess_eval_example(raw_data, tokenzier, args, ranking_dict=ranking_dict)

        if proc_data is not None:
            proc_dataset.append(proc_data)
    print(len(proc_dataset))
    data['data'] = proc_dataset
    return data

# train with ground truth supporting paragraphs
def process_train():
    args = _parse_args()
    tokenzier = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    dataset_prefix = 'datasets/hotpot'
    train_fname = join(dataset_prefix, 'hotpot_train_v1.1.json')
    dev_fname = join(dataset_prefix, 'hotpot_dev_distractor_v1.json')
    
    train_set = preprocess_split(train_fname, tokenzier, args, is_train=True)
    dev_set = preprocess_split(dev_fname, tokenzier, args, is_train=True)
    
    output_prefix = 'outputs/'

    train_outfile = 'train_hpqa.json'
    dev_outfile = 'gtsp-dev_hpqa.json'
    if args.do_mini:
        train_outfile = 'mini-' + train_outfile
        dev_outfile = 'mini-' + dev_outfile
    train_outfile = join(output_prefix, train_outfile)
    dev_outfile = join(output_prefix, dev_outfile)

    dump_json(train_set, train_outfile)
    # dump_json(dev_set, dev_outfile)


def process_dev():
    args = _parse_args()
    tokenzier = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    dataset_prefix = 'datasets/hotpot'
    output_prefix = 'outputs/'
    dev_fname = join(dataset_prefix, 'hotpot_dev_distractor_v1.json')
    ranking_fname = join('misc/', 'dev_ranker_preds.bin')
    ranking_dict = load_bin(ranking_fname)

    dev_set = preprocess_split(dev_fname, tokenzier, args, is_train=False, ranking_dict=ranking_dict)
    

    dev_outfile = 'dev_hpqa.json'
    if args.do_mini:
        dev_outfile = 'mini' + dev_outfile
    dev_outfile = join(output_prefix, dev_outfile)
    
    dump_json(dev_set, dev_outfile)


def process_hpqa_perturb(split):
    ids = read_hotpot_perturbations(split)

    dev_dataset = 'outputs/dev_hpqa.json'
    dataset = read_json(dev_dataset)
    
    type_data = []
    for ex in dataset['data']:
        question_id = ex['paragraphs'][0]['qas'][0]['id']
        if question_id in ids:
            type_data.append(ex)
    # print(bridge_type_data[0])
    # print(bridge_type_data[-1])

    type_data.sort(key=lambda x: x['paragraphs'][0]['qas'][0]['id'])        
    comp_dataset = {'version': '1.1', 'data': type_data}    
    dump_json(comp_dataset, f'outputs/{split}-perturb_hpqa.json', indent=2)
    
if __name__ == "__main__":
    process_train()
    process_dev()
    process_hpqa_perturb('yesno')
    process_hpqa_perturb('bridge')
