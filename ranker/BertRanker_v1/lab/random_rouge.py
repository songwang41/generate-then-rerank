"""

查看oracle和随机挑出来的结果的rouge

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


with open('../data/cnndm_debug/test/gen_from_16/all.json') as f:
    data = json.load(f)

scorer = compute_rouge()

refs = []
for d in data:
    refs.append(d['target_untok'])

rouge1_sum = 0
rouge2_sum = 0
rougeLsum_sum = 0

for _ in range(5):
    preds = []
    for d in data:
        p = random.sample(d['candidates_untok'], 1)[0][0]
        preds.append(p)

    metrics = scorer((preds, refs))
    rouge1_sum += metrics['rouge1']
    rouge2_sum += metrics['rouge2']
    rougeLsum_sum += metrics['rougeLsum']

print('=============================random===================================')
print(rouge1_sum/5)
print(rouge2_sum/5)
print(rougeLsum_sum/5)


preds = []
for d in data:
    p = d['candidates_untok'][0][0]
    preds.append(p)

metrics = scorer((preds, refs))
print('=============================oracle===================================')
print(metrics['rouge1'])
print(metrics['rouge2'])
print(metrics['rougeLsum'])
