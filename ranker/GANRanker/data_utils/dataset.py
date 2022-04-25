from lib2to3.pgen2 import token
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import pandas as pd
import json
import copy
import numpy as np
import random
from pandas import DataFrame
import queue
import os
import pickle


class ReRankingDataset(Dataset):

    def __init__(self, dataset_name_or_path, split = 'train', generator_tokenizer = None, reranker_tokenizer = None, args = None, shuffle=False, is_train = False, only_predict = False):

        self.args = args
        self.only_predict = only_predict
        self.shuffle = shuffle
        self.isdir = os.path.isdir(dataset_name_or_path)
        self.args = args
        self.is_train = is_train
        if self.isdir:
            # directly input the data file dir
            self.fdir = dataset_name_or_path
        else:
            # input dataset name
            self.fdir = "data/%s"%(dataset_name_or_path)
        
        self.data = self.read(self.fdir, split, generator_tokenizer, reranker_tokenizer)
        self.len = len(self.data)

    def read(self, fdir, split, generator_tokenizer, reranker_tokenizer):
        if os.path.exists('%s/%s_data.pkl'%(fdir, split)) and self.args.load_tokenized_data:
            # print('load preprossed dataset from ../data/%s/%s_data.pkl'%(dataset_name, split))
            samples = pickle.load(open('%s/%s_data.pkl'%(fdir, split),'rb'))
        else:
            with open('%s/%s_data.json'%(fdir, split), encoding='utf-8') as f:
                raw_data = json.load(f)
            # process dialogue
            samples = []
            for d in raw_data:
                content_ids = generator_tokenizer.convert_tokens_to_ids(generator_tokenizer.tokenize(d['source']))
                content_ids = content_ids[:self.args.generator_max_source_length]
                content_ids =  [generator_tokenizer.bos_token_id] + content_ids + [generator_tokenizer.eos_token_id]

                if self.only_predict:
                    target_ids = None
                else:
                    target_ids = generator_tokenizer.convert_tokens_to_ids(generator_tokenizer.tokenize(d['target']))
                    target_ids = target_ids[:self.args.max_target_length]
                    target_ids = [generator_tokenizer.bos_token_id] + target_ids + [generator_tokenizer.eos_token_id]

                if self.only_predict and 'oracle' not in d.keys():
                    oracle_ids = None
                else:
                    oracle_ids = generator_tokenizer.convert_tokens_to_ids(oracle_ids.tokenize(d['oracle']))
                    oracle_ids = oracle_ids[:self.args.max_target_length]
                    oracle_ids = [generator_tokenizer.bos_token_id] + oracle_ids + [generator_tokenizer.eos_token_id]

                if not self.is_train and not self.args.generate_eval_candidates:
                    # for evaluation and prediction, we need to create the candidate ids for the reranker
                    candidates_ids = []
                    source_ids = reranker_tokenizer.convert_tokens_to_ids(reranker_tokenizer.tokenize(d['source']))[:self.args.reranker_max_source_length]
                    source_ids = [reranker_tokenizer.bos_token_id] + source_ids + [reranker_tokenizer.sep_token_id]
                    for c in d['candidates']:
                        c_ids = reranker_tokenizer.convert_tokens_to_ids(reranker_tokenizer.tokenize(c))[:self.args.max_target_length]
                        candidates_ids.append(source_ids + c_ids)


                samples.append({
                    'content_ids': content_ids, #content_ids[self.args.max_sent_len:],
                    'target_ids': target_ids,
                    'oracle_ids': oracle_ids,
                    'target_text': d['target'] if not self.only_predict else None,
                    'source_text': d['source'],
                    'oracle_text': d['oracle'] if not self.only_predict and 'oracle' not in d.keys() else None,
                    'candidates': d['candidates'] if not self.is_train and not self.args.generate_eval_candidates else None,
                    'candidates_ids': candidates_ids if not self.is_train and not self.args.generate_eval_candidates else None,
                })
        
        new_samples = []
        for d in samples:
            new_d = {}
            new_d['content_ids'] = d['content_ids']

            if self.args.oracle_as_label: # whether to use the oracle as the target to train generator in NLL
                if d['oracle_ids'] is None:
                    raise ValueError("use oracle as label but no oracle is provided")
                new_d['labels'] = d['oracle_ids']
            else:
                new_d['labels'] = d['target_ids']

            if self.args.pos_type == 'oracle': # whether to use the oracle as the positive sample for training the reranker
                new_d['pos_text'] = d['oracle_text'] 
            elif self.args.pos_type == 'groundtruth':
                new_d['pos_text'] = d['target_text']
            elif self.args.pos_type == 'generate':
                new_d['pos_text'] = None

            new_d['target_text'] = d['target_text'] # this is for picking the candidates generated by the generator, and the evaluation
            new_d['source_text'] = d['source_text']
            if self.is_train or self.args.generate_eval_candidates:
                new_d['candidates'] = None # candidate generated by the teacher model is not needed in training
                new_d['candidates_ids'] = None
            else:
                new_d['candidates'] = d['candidates']
                new_d['candidates_ids'] = d['candidates_ids']
            new_samples.append(new_d)

        if self.is_train and self.shuffle:
            random.shuffle(new_samples)
        return new_samples

    def __getitem__(self, index):
        '''

        :param index:
        :return:
            text_ids:
            token_types:
            label
        '''
        
        return {
            'input_ids': torch.LongTensor(self.data[index]['content_ids']),
            'labels': torch.LongTensor(self.data[index]['labels']) if self.data[index]['labels'] is not None else None,
            'pos_text': self.data[index]['pos_text'],
            'target_text': self.data[index]['target_text'],
            'source_text': self.data[index]['source_text'],
            'candidates': self.data[index]['candidates'],
            'candidates_ids': self.data[index]['candidates_ids']
        }


    def __len__(self):
        return self.len

    # def collate_fn(self, data):
    #     '''

    #     :param data:
    #         content_ids
    #         token_types
    #         labels

    #     :return:

    #     '''
    #     content_ids = pad_sequence([d[0] for d in data], batch_first = True, padding_value = 1) # (B, T, )
    #     labels = pad_sequence([d[1] for d in data], batch_first = True, padding_value=-100)



    #     attention_mask = pad_sequence([d[2] for d in data], batch_first = True)

    #     sample = {}
    #     sample['input_ids'] = content_ids
    #     sample['labels'] = labels
    #     sample['attention_mask'] = attention_mask
    #     # print(sample)
    #     # exit()
    #     return sample
