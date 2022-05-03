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
from data_utils.metric_utils import compute_rouge, compute_qg, compute_coqa, compute_dialog


nltk.download('punkt')
nltk.download("omw-1.4", quiet=True)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--prediction_dir", type=str)
#parser.add_argument("--golden", type=str, help="Gold output file.")
args = parser.parse_args()


def read_data(data_dir):
    with open(os.path.join(data_dir, 'test_data.json')) as f:
        data = json.load(f)
    refs = []
    for d in data:
        refs.append(d['target'])

    return refs

def read_prediction(prediciton_dir):
    preds = []
    with open(os.path.join(prediciton_dir)) as f:
        for line in f.readlines():
            preds.append(line.strip())

    return preds

if __name__ == "__main__":
    refs = read_data(args.data_dir)
    preds = read_prediction(args.prediction_dir)
    print('==============================================================')
    print(args.dataset_name)
    print(args.prediction_dir)

    compute_metrics = None
    if args.dataset_name in ["cnndm", "cnndm_debug", "samsum", "xsum"]:
        compute_metrics = compute_rouge()
    elif args.dataset_name in ['squadqg']:
        compute_metrics = compute_qg()
    elif args.dataset_name in ['coqa']:
        compute_metrics = compute_coqa()
    elif args.dataset_name in ['personachat']:
        compute_metrics = compute_dialog()

    assert compute_metrics is not None

    print(compute_metrics((preds,refs), decoded=True))
