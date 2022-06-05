import json
import os
import argparse
from os import write
from weakref import ref
from tqdm import tqdm
import queue
import numpy as np
import random
import pickle
import time
import nltk
from data_utils.metric_utils import compute_code_gen
from data_utils.bleu import _bleu


nltk.download('punkt')
nltk.download("omw-1.4", quiet=True)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--prediction_dir", type=str)
parser.add_argument("--ref_file_type", type=str)
#parser.add_argument("--golden", type=str, help="Gold output file.")
args = parser.parse_args()


def read_data(data_dir):
    refs = []
    with open(os.path.join(data_dir, 'test.json'), encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line=line.strip()
            d=json.loads(line)
            refs.append(d['code'])
    return refs

def read_prediction(prediciton_dir):
    preds = []
    with open(os.path.join(prediciton_dir)) as f:
        for line in f.readlines():
            preds.append(line.strip())

    return preds

if __name__ == "__main__":
    if args.ref_file_type == 'json':
        refs = read_data(args.data_dir)

        preds = read_prediction(args.prediction_dir)
        print('==============================================================')
        print(args.dataset_name)
        print(args.prediction_dir)

        compute_metrics = compute_code_gen()
        assert compute_metrics is not None

        print(compute_metrics((preds,refs)))
    elif args.ref_file_type == 'txt':
        bleu_score = _bleu(args.prediction_dir, args.data_dir)

        refs = read_prediction(args.data_dir)
        preds = read_prediction(args.prediction_dir)

        EM = [] 
        for p,t in zip(preds, refs):
            EM.append(p.split()==t.split())
        em_score = np.mean(EM) * 100

        print({'bleu':bleu_score, 'em': em_score})

