from data_utils.metric_utils import compute_coqa, compute_dialog, compute_rouge, compute_qg
import json
import nltk

# /weizhou_data/generation/ranker/GANRanker/data/glge/squadqg
with open('data/glge/squadqg/test_data.json') as f:
    data = json.load(f)

compute = compute_dialog()

preds = [d['target'] for d in data]
refs = [d['target'] for d in data]
result = (preds, refs)
print(len(preds))

print(compute(result))

# print(compute)

# preds = [nltk.word_tokenize(pred) for pred in preds]
# labels = [nltk.word_tokenize(label) for label in refs]

# print(compute_bleu(labels, preds, max_order=1, smooth=True))