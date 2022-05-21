import json
from rouge_score import rouge_scorer, scoring
scorer = rouge_scorer.RougeScorer(rouge_types = ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
import matplotlib.pyplot as plt
import argparse
import numpy as np
import nltk
from tqdm import tqdm
n = 16

nltk.download('punkt')
nltk.download("omw-1.4", quiet=True)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
# parser.add_argument("--dataset_name", type=str)
parser.add_argument("--pred_dir", type=str)
parser.add_argument("--save_dir", type=str)
#parser.add_argument("--golden", type=str, help="Gold output file.")
args = parser.parse_args()


with open(args.data_dir, encoding='utf-8') as f:
    data = json.load(f)

targets = []
for d in data:
    targets.append(d['target'])

preds = []
with open(args.pred_dir,encoding='utf-8') as f:
    for line in f.readlines():
       preds.append(line.strip())


preds = [pred.strip() for pred in preds]
targets = [target.strip() for target in targets]

preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

scores = []
for i in tqdm(range(len(targets))):
    t = targets[i]
    scores = []
    ps = preds[i * 16: (i+1)*16]
    score_this = []
    for j,p in enumerate(ps):
        s = scorer.score(t, p)
        score_this.append(s["rouge1"].fmeasure)

    scores.append(score_this)

scores = np.array(scores)

min_scores = np.min(scores, axis=1)
max_scores = np.max(scores, axis=1)

results = {
    'min_scores': min_scores.tolist(),
    'max_scores': max_scores.tolist()
}

with open(args.save_dir, 'w', encoding='utf-8') as f:
    json.dump(results, f)
# scores = np.array(scores)

# min_scores = np.min(scores, axis=1)
# max_scores = np.max(scores, axis=1)