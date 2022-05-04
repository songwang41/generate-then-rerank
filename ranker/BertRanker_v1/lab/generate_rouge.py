"""

查看模型生成的结果的rouge

"""

from dataclasses import dataclass
import random
import json
from dataclasses import dataclass
from datasets import  load_metric
import nltk

class compute_rouge:
    def __init__(self):
        self.metric = load_metric('rouge')

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def __call__(self, eval_preds):
        preds, labels = eval_preds

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(preds, labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

with open('../../data/cnndm/test_data.json', encoding='utf-8') as f:
    raw_data = json.load(f)

candidates = []
with open("/weizhou_data/generation/bart-samsum/bart-base-diverse/saves/cnndm/generated_predictions.txt" , encoding='utf-8') as f:
    for line in f.readlines():
        candidates.append(line.strip())

refs = []
preds = candidates[:5000]
cur_line = 0
for d in raw_data[:5000]:
    refs.append(d['target'])

scorer = compute_rouge()
metrics = scorer((preds, refs))
print(metrics['rouge1'])
print(metrics['rouge2'])
print(metrics['rougeLsum'])
