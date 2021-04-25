import os
import sys
import argparse
sys.path.append('.')
from os.path import join
from eval_hotpot_exp.yesno_testers import YesNoAtAttrTester, YesNoLAtAttrTester, YesNoTokIGTester, YesNoArchTester, YesNoShapTester, YesNoLimeTester
from eval_hotpot_exp.utils import HotpotPredictor, get_oringinal_prediction, make_qa_data, get_prediction_confidence
from common.interp_utils import interp_metrics
from common.dataset_utils import read_hotpot_perturbations


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--gpu_id', type=str, default='0')
    args = parser.parse_args()
    return args

def perturb_target_properties(meta, predictor, orig_prediction, verbose=False):
    template_info = meta['perturb_property']
    par_templates = template_info['context_templates']
    property_candidates = template_info['property_candidates']

    question = meta['original']['question']

    perturbations = []
    for p1_idx, p1 in enumerate(property_candidates):
        for p2_idx, p2 in enumerate(property_candidates):
            filled_pars = [x.replace('#PROPERTY', y) for (x,y) in zip(par_templates, [p1, p2])]
            context = ' '.join(filled_pars)
            answer = 'null'
            id = f'{p1_idx}_{p2_idx}'

            perturbations.append(make_qa_data(context, question, answer, id))    
    predictions = predictor.predict(perturbations)

    status = len(set(predictions.values())) != 1    
    if verbose:
        print('Original', orig_prediction)
        print(predictions)

    return (1 if status else 0)

def verify_example(qas_id, annotation, tester):
    interp = tester.load_interp(qas_id)    
    factor = tester.get_impacts_of_property(interp, annotation)
    # print(factor)
    return factor

def get_property_label(qas_id, meta, predictor, verbose=False):        
    orig_prediction = get_oringinal_prediction(meta, predictor)
    return perturb_target_properties(meta, predictor, orig_prediction)

def main():
    args = _parse_args()
    predictor = HotpotPredictor(gpu=args.gpu_id)
    if args.method == 'conf':
        pass
    elif args.method == 'tokig':
        tester = YesNoTokIGTester()
    elif args.method == 'latattr':
        tester = YesNoLAtAttrTester()
    elif args.method == 'atattr':
        tester = YesNoAtAttrTester()
    elif args.method == 'arch':
        tester = YesNoArchTester()
    elif args.method == 'shap':
        tester = YesNoShapTester()
    elif args.method == 'lime':
        tester = YesNoLimeTester()
    else:
        raise RuntimeError('No such interp method')
    annotation_dict = read_hotpot_perturbations('yesno')
    factors = []
    labels = []
    for qid, meta in annotation_dict.items():
        prop_label = get_property_label(qid, meta, predictor)
        if args.method == 'conf':
            factor = get_prediction_confidence(meta, predictor)
        else:
            factor = verify_example(qid, meta, tester)
        print(qid, prop_label, factor)
        labels.append(prop_label)
        factors.append(factor)
    interp_metrics(factors, labels)
main()