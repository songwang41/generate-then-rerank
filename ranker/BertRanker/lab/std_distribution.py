"""

查看数据集中每个候选rouge的标准差

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


rouge1_std = []
rouge2_std = []
rougeLsum_std = []
similarity_std = []
for d in tqdm(data):
    rouge1_list = []
    rouge2_list = []
    rougeLsum_list = []
    similarity_list = []
    ref = "\n".join(nltk.sent_tokenize(d['target_untok']))
    for i in range(16):
        pred = d['candidates_untok'][i][0]
        pred = "\n".join(nltk.sent_tokenize(pred))
        score = scorer.score(ref, pred)
        rouge1_list.append(score["rouge1"].fmeasure * 100)
        rouge2_list.append(score["rouge2"].fmeasure * 100)
        rougeLsum_list.append(score["rougeLsum"].fmeasure * 100)
        similarity_list.append((score["rouge1"].fmeasure / 0.45 + score["rouge2"].fmeasure / 0.2 + score["rougeLsum"].fmeasure / 0.4)/3)

    rouge1_std.append(np.std(rouge1_list))
    rouge2_std.append(np.std(rouge2_list))
    rougeLsum_std.append(np.std(rougeLsum_list))
    similarity_std.append(np.std(similarity_list))

results = {
    'rouge1_std':rouge1_std,
    'rouge2_std':rouge2_std,
    'rougeLsum_std':rougeLsum_std,
    'similarity_std':similarity_std
}

with open('lab_statis/test_std_distribution.json', 'w', encoding="utf-8") as f:
    json.dump(results,f)