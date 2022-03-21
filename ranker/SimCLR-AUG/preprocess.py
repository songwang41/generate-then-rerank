import json
from rouge_score import rouge_scorer, scoring
from transformers import RobertaTokenizer
import nltk
import os
from tqdm import tqdm



def generate_data(dataset_name, split, tokenizer, sent_detector, num_cand, eval_metric):
    '''
    return: json files, each file contains
        {
            "source":".....",
            "target":".....",
            "candidates":[
                {
                    "text":".....",
                    "score": "....."
                }
            ]
        }
    '''
    
    if not os.path.exists('data/%s/%s/gen_from_%d'%(dataset_name,split, num_cand)):
        os.makedirs('data/%s/%s/gen_from_%d'%(dataset_name,split, num_cand))
    if dataset_name == 'samsum':
        rouge_types = ["rouge1", "rouge2", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)

        with open('../data/samsum/%s_data.json'%(split), encoding='utf-8') as f:
            raw_data = json.load(f)
        candidates = []
        with open('../data/samsum/candidates/%s/generated_predictions_%d.txt'%(split, num_cand), encoding='utf-8') as f:
            for line in f.readlines():
                candidates.append(line.strip())
        cur_line = 0
        samples = []
        i = 0
        for d in tqdm(raw_data):
            article = ''
            for u in d['dialogue']:
                article += u['text']
                article += ' '
                article += tokenizer.sep_token
                article += ' '
            article_tok = tokenizer.tokenize(article)
            target = d['summary']
            target_tok = tokenizer.tokenize(target)
            target_for_score = '\n'.join(sent_detector.tokenize(target))
            cand_list = []
            cand_list_tok = []
            for _ in range(num_cand):
                cand = candidates[cur_line]
                cand_for_score = '\n'.join(sent_detector.tokenize(cand))
                score = scorer.score(target_for_score, cand_for_score)
                similarity = (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
                cand_list.append((cand, similarity))
                cand_list_tok.append((tokenizer.tokenize(cand), similarity))
                cur_line += 1
            
            cand_list = sorted(cand_list, key=lambda x:x[1], reverse=True)
            cand_list_tok = sorted(cand_list_tok, key=lambda x:x[1], reverse=True)

            sample = {
                'source': article_tok,
                'source_untok': article,
                'target': target_tok,
                'target_untok': target,
                'candidates': cand_list_tok,
                'candidates_untok': cand_list
            }
            samples.append(sample)


            with open('data/%s/%s/gen_from_%d/%d.json'%(dataset_name, split, num_cand, i), 'w', encoding='utf-8') as f:
                json.dump(sample, f)
            i += 1
        with open('data/%s/%s/gen_from_%d/all.json'%(dataset_name, split, num_cand), 'w', encoding='utf-8') as f:
                json.dump(samples, f)
        



                



if __name__ == "__main__":
    dataset_name = 'samsum'
    num_cand = 40
    eval_metric = 'rouge'
    tokenizer = RobertaTokenizer.from_pretrained('/weizhou_data/models/roberta-base')
    nltk.download('punkt')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    generate_data(dataset_name, 'train',  tokenizer, sent_detector, 32, eval_metric)
    generate_data(dataset_name, 'dev',  tokenizer, sent_detector, 32, eval_metric)
    for n in range(1, 41):
        print(n)
        generate_data(dataset_name, 'test', tokenizer, sent_detector, n, eval_metric)

    # for n in range(1, 32):
    #     generate_data(dataset_name, 'test', tokenizer, sent_detector, n, eval_metric)
 