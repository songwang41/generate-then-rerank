from dataclasses import dataclass
import torch
from rouge_score import rouge_scorer, scoring
from dataclasses import dataclass
from datasets import  load_metric
import nltk
import random

class compute_rouge:
    def __init__(self):
        self.metric = load_metric('rouge')
        self.scorer = rouge_scorer.RougeScorer(rouge_types = ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)

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


    def get_candidates(self, targets,  preds, num_cand, max_num, strategy):
        """
        args:
            targets: list of targets for each sample
            pos: list of positive samples for each sample
            preds: list of predictions, length == len(targets) * num_cand
            num_cand: number of candidates
            max_num: number of returned indices per sample 
        returns:
            indices: Torch tensor, (B * (C-1), ), since the positive candidate is not generated from the generator
            candiates: candidates, with the length of len(targets) * max_num
            NOTE: We should always keep the positive sequences in the first candidate for each sample
        """
        preds_processed, targets_processed = self.postprocess_text(preds, targets)

        indices = []
        candidates = []
        rewards = []
        for i,t in enumerate(targets_processed):
            scores = []
            ps = preds_processed[i * num_cand: (i+1)*num_cand]
            for j,p in enumerate(ps):
                s = self.scorer.score(t, p)
                scores.append((j + i * num_cand, s["rouge1"].fmeasure / 0.45 + s["rouge2"].fmeasure / 0.2 + s["rougeLsum"].fmeasure / 0.4, p))

            scores = sorted(scores, key = lambda x: x[1], reverse=True)
            

            idx_this = [scores[0][0]] # the first as pos
            cand_this = [scores[0][2]]
            rewards_this = [scores[0][1]]
            scores = scores[1:]

            if strategy == 'random':
                s_for_pick = random.sample(scores, max_num - 1)
                idx_this +=  [s[0] for s in s_for_pick]
                cand_this +=  [s[2] for s in s_for_pick]
                rewards_this += [s[1] for s in s_for_pick]
            else:
                if strategy == 'top':
                    idx_this +=  [s[0] for s in scores[:max_num-1]]
                    cand_this +=  [s[2] for s in scores[:max_num-1]]
                    rewards_this += [s[1] for s in scores[:max_num-1]]
                elif strategy == 'bottom':
                    idx_this +=  [s[0] for s in scores[-max_num+1:]]
                    cand_this +=  [s[2] for s in scores[-max_num+1:]]
                    rewards_this += [s[1] for s in scores[-max_num+1:]]
                elif strategy == 'top-bottom':
                    n_top = (max_num-1) // 2
                    n_bottom = (max_num-1) - n_top
                    idx_this +=  [s[0] for s in scores[:n_top]]
                    cand_this += [s[2] for s in scores[:n_top]]
                    idx_this +=  [s[0] for s in scores[-n_bottom:]]
                    cand_this += [s[2] for s in scores[-n_bottom:]]
                    rewards_this += [s[1] for s in scores[:n_top]]
                    rewards_this += [s[1] for s in scores[-n_bottom:]]


            indices += idx_this
            candidates += cand_this
            rewards.append(rewards_this)
        
        return torch.LongTensor(indices), candidates, torch.FloatTensor(rewards)

    
    def get_reward(self, targets,  preds):
        """
        args:
            targets: list of targets for each sample
            preds: list of predictions, length == len(targets) * num_cand
        returns:
            rewards: the scores
            NOTE: We should always keep the positive sequences in the first candidate for each sample
        """
        num_cand = len(preds)//len(targets)
        preds_processed, targets_processed = self.postprocess_text(preds, targets)

        rewards = []
        for i,t in enumerate(targets_processed):
            scores = []
            ps = preds_processed[i * num_cand: (i+1)*num_cand]
            for j,p in enumerate(ps):
                s = self.scorer.score(t, p)
                scores.append(s["rouge1"].fmeasure / 0.45 + s["rouge2"].fmeasure / 0.2 + s["rougeLsum"].fmeasure / 0.4)

            rewards += scores
        
        return torch.FloatTensor(rewards)



