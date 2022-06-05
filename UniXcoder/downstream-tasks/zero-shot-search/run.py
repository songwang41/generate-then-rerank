# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import sys 


import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from tqdm import tqdm
from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)  
import re
from io import StringIO
import  tokenize

logger = logging.getLogger(__name__)


def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 index,
                 label

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.index = index
        self.label = label

        
def convert_examples_to_features(js,tokenizer,args,lang):
    """convert examples to token ids"""
    if "func" in js:
        code = " ".join(remove_comments_and_docstrings(js['func'],lang).split())
    else:
        code = " ".join(remove_comments_and_docstrings(js['code'],lang).split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    return InputFeatures(code_tokens,code_ids,js["index"],int(js['label']))

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, lang):
        self.examples = []
        data = []
        with open(file_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                data.append(js)

        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args,lang))

        for idx, example in enumerate(self.examples[:1]):
            logger.info("*** Example ***")
            logger.info("label: {}".format(example.label))
            logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
            logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                
        self.label_examples={}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label]=[]
            self.label_examples[e.label].append(e)                           
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):          
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].label))
            

    
def evaluate(args, model, tokenizer, file_name, candidate_file_name):
    query_dataset = TextDataset(tokenizer, args, file_name, args.query_lang)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    candidate_dataset = TextDataset(tokenizer, args, candidate_file_name, args.candidate_lang)
    candidate_sampler = SequentialSampler(candidate_dataset)
    candidate_dataloader = DataLoader(candidate_dataset, sampler=candidate_sampler, batch_size=args.eval_batch_size, num_workers=4)    

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num Query = %d", len(query_dataset))
    logger.info("  Num Candidate = %d", len(candidate_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    query_vecs = [] 
    query_labels = []
    candidate_vecs = []
    candidate_labels = []
    # Obtain query vectors
    for batch in query_dataloader:  
        code_inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs) 
            query_vecs.append(code_vec.cpu().numpy()) 
            query_labels.append(label.cpu().numpy())
    
    # Obtain candidate vectors
    for batch in candidate_dataloader:  
        code_inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs) 
            candidate_vecs.append(code_vec.cpu().numpy()) 
            candidate_labels.append(label.cpu().numpy())
            
    model.train() 

    # Calculate cosine score
    query_vecs = np.concatenate(query_vecs,0)
    candidate_vecs = np.concatenate(candidate_vecs,0)
    query_labels = list(np.concatenate(query_labels,0))
    candidate_labels = list(np.concatenate(candidate_labels,0))
    candidate_indexs =[candidate_dataset.examples[i].index for i in range(len(candidate_dataset))]
    query_indexs = [query_dataset.examples[i].index for i in range(len(query_dataset))]
    scores = np.matmul(query_vecs,candidate_vecs.T)
    
    # Calculate MAP score
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    MAP=[]
    results = {}
    for i in range(scores.shape[0]):
        cont=0
        label=int(query_labels[i])
        query_index = query_indexs[i]
        results[query_index] = [label,candidate_labels[sort_ids[i][0]],candidate_indexs[sort_ids[i][0]]]
        Avep = []
        for j,index in enumerate(list(sort_ids[i])):
            if query_index==candidate_indexs[index]:
                cont+=1
                continue
            if  int(candidate_labels[index])==label:
                Avep.append((len(Avep)+1)/(j+1-cont))
        if len(Avep)!=0:
            MAP.append(sum(Avep)/len(Avep))
   
    result = {
        "eval_map":float(np.mean(MAP))
    }
    return result

                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--query_data_file", default=None, type=str, required=False,
                        help="The input training data file (a json file).")
    parser.add_argument("--candidate_data_file", default=None, type=str, required=False,
                        help="The input training data file (a json file).")    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    
    parser.add_argument("--query_lang", default=None, type=str, required=False,
                        help="Programming language of query.")    
    parser.add_argument("--candidate_lang", default=None, type=str,
                        help="Programming language of candidate.")
    
    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")



    #print arguments
    args = parser.parse_args()
      
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 
 
    
    model=Model(model)
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model) 
        

    result=evaluate(args, model, tokenizer,args.query_data_file,args.candidate_data_file)
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key]*100,2)))



if __name__ == "__main__":
    main()


