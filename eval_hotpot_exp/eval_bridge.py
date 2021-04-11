
import os
import sys
sys.path.append('.')
from os.path import join
import argparse

from eval_hotpot_exp.bridge_testers import BridgeArchTester, BridgeTokIGTester, BridgeAtAttrTester, BridgeLAtAttrTester
from eval_hotpot_exp.utils import HotpotPredictor, get_oringinal_prediction, make_qa_data
from common.interp_utils import interp_metrics
from common.dataset_utils import read_hotpot_perturbations

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--gpu_id', type=str, default='0')
    args = parser.parse_args()
    return args

def perturb_adv_sentence(meta, predictor, orig_prediction):
    data = {}
    data['id'] = 'adv_sent'
    data['question'] = meta['original']['question']
    data['answer'] = meta['original']['answer']
    pars = [meta['adv_sent']['distractor'], meta['adv_sent']['contexts'][0], meta['adv_sent']['contexts'][1]]
    
    data['context'] = ' '.join(pars)

    prediction = predictor.predict([data])
    prediction = prediction['adv_sent']
    
    pars = [meta['adv_sent']['distractor1'], meta['adv_sent']['contexts'][0], meta['adv_sent']['contexts'][1]]
    data['context'] = ' '.join(pars)

    prediction1 = predictor.predict([data])
    prediction1 = prediction1['adv_sent']
        
    status = (prediction != orig_prediction) or (prediction1 != orig_prediction)
    return (0 if status else 1)


def get_adv_label(qas_id, meta, predictor, verbose=False):        
    orig_prediction = get_oringinal_prediction(meta, predictor)
    return perturb_adv_sentence(meta, predictor, orig_prediction)

def verify_example(qas_id, annotation, tester):
    interp = tester.load_interp(qas_id)
    val = -tester.get_impacts_of_primary_question(interp, annotation)    
    return val
    
def main():
    args = _parse_args()
    predictor = HotpotPredictor(gpu=args.gpu_id)
    if args.method == 'conf':
        pass
    elif args.method == 'tokig':
        tester = BridgeTokIGTester()
    elif args.method == 'latattr':
        tester = BridgeLAtAttrTester()
    elif args.method == 'atattr':
        tester = BridgeAtAttrTester()
    elif args.method == 'arch':
        pass
    else:
        raise RuntimeError('No such interp method')
    annotation_dict = read_hotpot_perturbations('bridge')
    factors = []
    labels = []
    for i, meta in annotation_dict.items():
        adv_label = get_adv_label(i, meta, predictor)
        factor = verify_example(i, meta, tester)
        print(meta['quick_id'], adv_label, factor)
        labels.append(adv_label)        
        factors.append(factor)
    interp_metrics(factors, labels)

main()