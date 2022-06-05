"""

查看数据集中每个候选rouge的分布

"""

import json
import numpy as np
from tqdm import tqdm

from dataclasses import dataclass
import random
import json
from dataclasses import dataclass
from datasets import  load_metric
import nltk

from rouge_score import rouge_scorer, scoring

rouge_types =  ["rouge1", "rouge2", "rougeLsum"]
scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)


with open("../data/cnndm_debug/test/gen_from_16/all.json", encoding='utf-8') as f:
    data = json.load(f)


rouge1_list = []
rouge2_list = []
rougeLsum_list = []
similarity_list = []
for d in tqdm(data):
    ref = "\n".join(nltk.sent_tokenize(d['target_untok']))
    for i in range(16):
        pred = d['candidates_untok'][i][0]
        pred = "\n".join(nltk.sent_tokenize(pred))
        score = scorer.score(ref, pred)
        rouge1_list.append(score["rouge1"].fmeasure)
        rouge2_list.append(score["rouge2"].fmeasure)
        rougeLsum_list.append(score["rougeLsum"].fmeasure)
        similarity_list.append(score["rouge1"].fmeasure / 0.45 + score["rouge2"].fmeasure / 0.2 + score["rougeLsum"].fmeasure / 0.4)

results = {
    'rouge1':rouge1_list,
    'rouge2':rouge2_list,
    'rougeLsum':rougeLsum_list,
    'similarity':similarity_list
}

with open('lab_statis/test_rouge_distribution.json', 'w', encoding="utf-8") as f:
    json.dump(results,f)