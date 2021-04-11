# EvalQAExpl
Code and data for [Evaluating Explanations for Reading Comprehension with Realistic Counterfactuals](https://arxiv.org/pdf/2104.04515.pdf).

![Aggregated attention attributions for a Hotpot Yes-No question.](https://github.com/xiye17/EvalQAExpl/docs/example_latattr.jpg)

## Requirements
* python==3.8
* huggingface-transformers==3.3.1
* pytorch==1.6.0

Please fix the version of huggingface-transformers, since the API changes pretty quickly.


## Experiments on HotpotQA (Yes-No and Brdige)

**Annotations for Counterfactuals**

Our annotations for hotpot counterfactuals can be found in `hotpot_counterfactuals` (50 for Bridge Type questions and 50 for Yes-No Type questions).

#### 1. Train Base HotpotQA Model (Distractor Settings)
Our HotpotQA base model is a select-and-answer style model. We first select 2 paragraphs using a document ranker model (a roberta classification model) to construct the context, and then use a roberta qa model to answer the question. Here, we only train the qa part and use provided scores to select paragraphs.  Detailed implmentation of the document ranker can be found in [NaiveHotpotBaseline](https://github.com/xiye17/NaiveHotpotBaseline).

a. put the [HotpotQA](https://hotpotqa.github.io/) in directory `datasets/hotpot`.

b. run `python make_hpqa_dataset.py`. This will generate squad-style dataset files in `outputs` direcotry. Naming convention follows `[split]-[dataset].json`, where `[dataset]` is `hpqa`. For training split, we use the *gold paragraphs* to train the model, for dev/bridge-perturb/yesno-perturb split we used paragraphs selected using the ranker provided outputs in `misc/dev_ranker_preds.bin`. Please refer this implemention ([NaiveHotpotBaseline](https://github.com/xiye17/NaiveHotpotBaseline)) for details.

c. run `sh run_qa.sh train hpqa [exp_id]` to train the roberta model. This will write all outputs to `exps/[dataset]-[exp_id]`. Please refer to the script for details of hyper-parameters.

d. store the trained model at `checkpoints/hpqa_roberta-base`. Create this foler and copy the needed files (`pytorch_model.bin`, `config.json`, and etc.) there.

e. verify the trained model using `sh run_qa.sh eval hpqa dev`. It should be able to achieve an exact match of 63% and f1 score os 77.0, comparable to strong single vanilla-bert-based model.


#### 2. Generate Interprations
We support the following explanation `[method]`s. (Experiments DiffMask are implemented in a separate repository).
* tokig: (token-level) [Integrated Gradient (Sundararajan et al., 2017)](https://arxiv.org/abs/1703.01365)
* arch: [Archipelago (Tsang et al., 2020)]()
* atattr:  [Attention Attribution (Hao et al., 2021)](https://arxiv.org/abs/2004.11207)
* latattr: layerwise attention attribution (Ours)


a. run `sh run_interp.sh hpqa [method] run [split]` to generate interpretation files for `outputs/[split]-hpqa.json` using `[method]`. For the following experiments we only need to generate interpretations files for yesno-perturb and bridge-perturb. E.g, `sh run_interp.sh hpqa latattr run yesno-perturb`. Note that interaction-based methods can take a long time. The generated interpretation files are `bin` files stored at `interpretations/[method]/[dataset]-[split]-[model]` directory, which can be loaded by `torch.load`.

b. (optional) we support visualizing the interpretations for `[tokig, atattr, latattr]` by runing `sh run_interp.sh hpqa [method] vis [split]`. This will generate figures stored at `visualizations/[method]/[dataset]-[split]-[model]`. (See the example in the beggining.)


#### 3. Evaluate Explanations with Counterfactuals
...
