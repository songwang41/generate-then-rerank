from dataclasses import dataclass
import torch
import numpy as np
from collections import Counter
from rouge_score import rouge_scorer, scoring
from dataclasses import dataclass
from datasets import  load_metric
import nltk
from .bleu import compute_bleu
import random


class compute_code_gen:
    def __init__(self):
        self.max_order = 4
        self.smooth = True

    def postprocess_text(self, preds, labels):
        preds = [pred.strip().split() for pred in preds]

        ref_files = [labels]
        reference_text = []
        for reference_file in ref_files:
            reference_text.append(reference_file)
            per_segment_references = []
            for references in zip(*reference_text):
                reference_list = []
                for reference in references:
                    reference_list.append(reference.strip().split())
                per_segment_references.append(reference_list)
        # labels = [label.strip().split() for label in labels]

        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, per_segment_references
    
    def compute_em(self, preds, targets):
        EM = [] 
        for p,t in zip(preds, targets):
            EM.append(p.strip().split()==t.strip().split())
        return np.mean(EM)

    def __call__(self, eval_preds):
        preds, labels = eval_preds

        # Some simple post-processing
        processed_preds, processed_labels = self.postprocess_text(preds, labels)

        bleu_score, _, _, _, _, _ = compute_bleu(processed_labels, processed_preds, self.max_order, self.smooth)
        em = self.compute_em(preds, labels)

        # Extract a few results from ROUGE
        result = {'bleu': bleu_score*100,
                  'em': em*100}

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
            ps_nopro = preds[i * num_cand: (i+1)*num_cand] # for the candidates, we use no preprocessed version
            for j,p in enumerate(ps):
                bleu_score, _, _, _, _, _ = compute_bleu([t], [p], self.max_order, self.smooth)
                em = self.compute_em(ps_nopro[j], targets[i])
                scores.append((j + i * num_cand, em / 0.33 + bleu_score / 0.67, ps_nopro[j].strip()))

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
            ps_nopro = preds[i * num_cand: (i+1)*num_cand] # for the candidates, we use no preprocessed version
            for j,p in enumerate(ps):
                bleu_score, _, _, _, _, _ = compute_bleu([t], [p], self.max_order, self.smooth)
                em = self.compute_em(ps_nopro[j], targets[i])
                scores.append(em / 0.33 + bleu_score / 0.67)

            rewards += scores
        
        return torch.FloatTensor(rewards)

