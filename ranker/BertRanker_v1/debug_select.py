import json
from data_utils.metric_utils import compute_rouge

compute_metrics = compute_rouge()
with open('data/cnndm-large/test/gen_from_16/all.json') as f:
    data = json.load(f)

targets = []
preds_best = []
preds_worst = []
for d in data:
    targets.append(d['target_untok'])
    preds_best.append(d['candidates_untok'][0])
    preds_worst.append(d['candidates_untok'][-1])

for p,t in zip(preds_best, targets):
    print(p)
    print(t)
    print('==========\n')

print(compute_metrics((preds_best, targets)))
print(compute_metrics((preds_worst, targets)))
