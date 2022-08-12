# this is for checking the results of each checkpoints
import json
import numpy as np
from tabnanny import check


results = []
best_metric = 0
best_check_point = 0
metric_for_best_model = 'generator_eval_rouge1'
model_dir = '/weizhou_data/generation/ranker/RLRanker_v7/saves/v7-large-cnndm-ncand8-npick8-top_pick'

check_points = list(range(1000, 13001, 1000))

for c in check_points:
    fdir = '%s/checkpoint-%d/beamsearch-16/eval_results.json'%(model_dir, c)
    with open(fdir) as f:
        r = json.load(f)
        if r[metric_for_best_model] > best_metric:
            best_metric = r[metric_for_best_model]
            best_check_point = c


fdir =  '%s/checkpoint-%d/beamsearch-16/predict_results.json'%(model_dir,best_check_point)


print('best check point', best_check_point)
with open(fdir) as f:
    r = json.load(f)
    for k,v in r.items():
        print(k, v)
    # print(r)