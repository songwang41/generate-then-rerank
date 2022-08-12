"""
    This code is for standardize the model predictions 
"""
import json
from rouge_score import rouge_scorer, scoring
from transformers import RobertaTokenizerFast, BartTokenizerFast
import nltk
import os
from tqdm import tqdm
import argparse
from data_utils.metric_utils import compute_coqa, compute_dialog, compute_qg, compute_rouge

nltk.download('punkt')
nltk.download("omw-1.4", quiet=True)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--file_type", type=str)

#parser.add_argument("--golden", type=str, help="Gold output file.")
args = parser.parse_args()

if args.file_type == 'txt':
    preds = []
    with open(args.data_dir) as f:
        for line in f.readlines():
            preds.append(line.strip())
elif args.file_type == 'json':
    preds = []
    with open(args.data_dir) as f:
        data = json.load(f)
    for d in data:
        preds.append(d['pred'])

preds_std = []
for p in preds:
    tokens = nltk.word_tokenize(p)
    new_tokens = []
    for t in tokens:
        if t == "'s":
            new_tokens += ["'", "s"]
        else:
            new_tokens.append(t)
    preds_std.append(' '.join(new_tokens))

with open(args.save_dir, 'w', encoding='utf-8') as f:
    for p in preds_std:
        f.write(p)
        f.write('\n')