import json
from rouge_score import rouge_scorer, scoring
from transformers import RobertaTokenizer
import nltk
import os
from tqdm import tqdm



def generate_data(dataset_name, candidate_dir, split, tokenizer, sent_detector, num_cand, eval_metric):
    '''
    args: 
    dataset_name
    candidate_dir: the file path for candidate files
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
    
    # if not os.path.exists('data/%s/%s/gen_from_%d'%(dataset_name,split, num_cand)):
    #     os.makedirs('data/%s/%s/gen_from_%d'%(dataset_name,split, num_cand))
    if dataset_name == 'samsum':
        rouge_types = ["rouge1", "rouge2", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)

        with open('../data/samsum/%s_data.json'%(split), encoding='utf-8') as f:
            raw_data = json.load(f)
        candidates = []
        with open(candidate_dir + '/%s/generated_predictions_%d.txt'%(split, num_cand), encoding='utf-8') as f:
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
                # similarity = (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
                similarity = score["rouge1"].fmeasure / 0.45 + score["rouge2"].fmeasure / 0.2 + score["rougeLsum"].fmeasure / 0.4
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


        #     with open('data/%s/%s/gen_from_%d/%d.json'%(dataset_name, split, num_cand, i), 'w', encoding='utf-8') as f:
        #         json.dump(sample, f)
        #     i += 1
        # with open('data/%s/%s/gen_from_%d/all.json'%(dataset_name, split, num_cand), 'w', encoding='utf-8') as f:
        #         json.dump(samples, f)

    elif dataset_name == 'cnndm':
        rouge_types = ["rouge1", "rouge2", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)

        with open('../data/cnndm/%s_data.json'%(split), encoding='utf-8') as f:
            raw_data = json.load(f)
        candidates = []
        with open(candidate_dir + '/%s/generated_predictions_%d.txt'%(split, num_cand), encoding='utf-8') as f:
            for line in f.readlines():
                candidates.append(line.strip())
        cur_line = 0
        samples = []
        i = 0
        for d in tqdm(raw_data):
            article = d['source']
            article_tok = tokenizer.tokenize(article)
            target = d['target']
            target_tok = tokenizer.tokenize(target)
            target_for_score = '\n'.join(sent_detector.tokenize(target))
            cand_list = []
            cand_list_tok = []
            for _ in range(num_cand):
                cand = candidates[cur_line]
                cand_for_score = '\n'.join(sent_detector.tokenize(cand))
                score = scorer.score(target_for_score, cand_for_score)
                # similarity = (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
                similarity = score["rouge1"].fmeasure / 0.45 + score["rouge2"].fmeasure / 0.2 + score["rougeLsum"].fmeasure / 0.4
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


        #     with open('data/%s/%s/gen_from_%d/%d.json'%(dataset_name, split, num_cand, i), 'w', encoding='utf-8') as f:
        #         json.dump(sample, f)
        #     i += 1
        # with open('data/%s/%s/gen_from_%d/all.json'%(dataset_name, split, num_cand), 'w', encoding='utf-8') as f:
        #         json.dump(samples, f)

    return samples
        



                



if __name__ == "__main__":
    dataset_name = 'cnndm'
    candidate_dir = "/weizhou_data/generation/bart-samsum/bart-base-diverse/results/cnndm"
    num_cand = 40
    eval_metric = 'rouge'
    tokenizer = RobertaTokenizer.from_pretrained('/weizhou_data/models/roberta-base')
    nltk.download('punkt')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    # generate_data(dataset_name, candidate_dir, 'train',  tokenizer, sent_detector, 16, eval_metric)
    # generate_data(dataset_name, candidate_dir, 'dev',  tokenizer, sent_detector, 16, eval_metric)
    # generate_data(dataset_name, candidate_dir, 'test', tokenizer, sent_detector, 16, eval_metric)
    # for n in range(2, 41):
    #     print(n)
    #     generate_data(dataset_name, candidate_dir, 'test', tokenizer, sent_detector, n, eval_metric)

    # for n in range(1, 32):
    #     generate_data(dataset_name, 'test', tokenizer, sent_detector, n, eval_metric)

    samples1 = generate_data(dataset_name, candidate_dir, 'train_1',  tokenizer, sent_detector, num_cand, eval_metric)
    samples2 = generate_data(dataset_name, candidate_dir, 'train_2',  tokenizer, sent_detector, num_cand, eval_metric)
    
    samples = samples1 + samples2

    if not os.path.exists('data/cnndm/train_half/gen_from_16'):
        os.makedirs('data/cnndm/train_half/gen_from_16')
    with open('data/cnndm/train_half/gen_from_16/all.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f)

    for i,s in enumerate(samples):
        with open('data/cnndm_debug/train_half/gen_from_16/%d.json'%(i), 'w', encoding='utf-8') as f:
            json.dump(s, f)


 