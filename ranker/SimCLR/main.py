import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import model
import pickle
import time
import numpy as np
import os
import json
import random
from rouge_score import rouge_scorer
from transformers import RobertaModel, RobertaTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp, ReRankingDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from model import RankingLoss
import math
import logging
import nltk
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)



def evaluation(args):
    # load data
    tok = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    test_set = ReRankingDataset("./data/%s/test/gen_from_%d"%(args.dataset, args.num_beams), tok, is_test=True, maxlen=512, 
                        is_sorted=False, maxnum=args.max_num, is_untok=True, cache_data = args.cache_data)
    dataloader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    # build models
    scorer = model.ReRanker(args.model_name_or_path, tok.pad_token_id)
    if args.cuda:
        scorer = scorer.cuda()
    scorer.load_state_dict(torch.load(os.path.join("./cache", args.state_dict_path), map_location=f'cuda:{args.gpuid[0]}'))
    scorer.eval()
    model_name = args.state_dict_path.split("/")[0]

    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    print(model_name)
    print(args.num_beams)
    mkdir("./result/%s"%model_name)
    mkdir("./result/%s/reference/rank_from_%d"%(model_name, args.num_beams))
    mkdir("./result/%s/candidate/rank_from_%d"%(model_name, args.num_beams))
    groundtruth_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    cnt = 0
    acc = 0
    scores = []
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, args.gpuid[0])
            samples = batch["data"]
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['target_score']
            similarity = similarity.cpu().numpy()
            max_ids = similarity.argmax(1)
            scores.extend(similarity.tolist())
            acc += (max_ids == batch["scores"].cpu().numpy().argmax(1)).sum()
            for j in range(similarity.shape[0]):
                sample = samples[j]
                prediction = sample["candidates"][max_ids[j]][0]
                score = groundtruth_scorer.score("\n".join(args.sent_detector.tokenize(sample["target_untok"])),
                                 "\n".join(args.sent_detector.tokenize(prediction)))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
                with open("./result/%s/candidate/rank_from_%d/%d.dec"%(model_name, args.num_beams, cnt), "w") as f:
                    f.write(prediction)
                with open("./result/%s/reference/rank_from_%d/%d.ref"%(model_name, args.num_beams, cnt), "w") as f:
                    f.write(sample["target_untok"])
                cnt += 1
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    print(f"accuracy: {acc / cnt}")
    print("rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))


def test(dataloader, scorer, args, gpuid):
    scorer.eval()
    loss = 0
    cnt = 0
    groundtruth_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            samples = batch["data"]
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['target_score'] # (B, C) (B,)
            similarity = similarity.cpu().numpy()
            # if i % 1000 == 0:
            #     print(f"test similarity: {similarity[0]}")
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):
                cnt += 1
                sample = samples[j]
                prediction = sample["candidates"][max_ids[j]][0]
                score = groundtruth_scorer.score("\n".join(args.sent_detector.tokenize(sample["target_untok"])),
                                 "\n".join(args.sent_detector.tokenize(prediction)))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    scorer.train()
    loss = 1 - ((rouge1 + rouge2 + rougeLsum) / 3)
    
    if len(args.gpuid) > 1:
        loss = torch.FloatTensor([loss]).to(gpuid)
        dist.all_reduce(loss, op=dist.reduce_op.SUM)
        loss = loss.item() / len(args.gpuid)
    return loss, rouge1, rouge2, rougeLsum


def run(rank, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        recorder = Recorder(args.log_name, args.log)
    tok = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    train_set = ReRankingDataset("./data/%s/train/gen_from_%d"%(args.dataset, args.num_beams), tok, maxlen=args.max_len, maxnum=args.max_num, cache_data = args.cache_data)
    val_set = ReRankingDataset("./data/%s/dev/gen_from_%d"%(args.dataset, args.num_beams), tok, is_test=True, maxlen=512, is_sorted=False, maxnum=args.max_num, cache_data = args.cache_data)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	 train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
    	 val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False,  collate_fn=collate_fn_val)
    # build models
    scorer = model.ReRanker( args.model_name_or_path, tok.pad_token_id) 
    if len(args.state_dict_path) > 0:
        scorer.load_state_dict(torch.load(os.path.join("./cache", args.state_dict_path), map_location=f'cuda:{gpuid}'))
    if args.cuda:
        if len(args.gpuid) == 1:
            scorer = scorer.cuda()
        else:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            scorer = nn.parallel.DistributedDataParallel(scorer.to(gpuid), [gpuid], find_unused_parameters=True)
    scorer.train()
    init_lr = args.max_lr / args.warmup_steps
    s_optimizer = optim.Adam(scorer.parameters(), lr=init_lr)
    if is_master:
        recorder.write_config(args, [scorer], __file__)
    minimum_loss = 100
    all_step_cnt = 0
    # start training
    for epoch in range(args.epoch):
        s_optimizer.zero_grad()
        step_cnt = 0
        sim_step = 0
        avg_loss = 0
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['target_score']
            loss = args.scale * RankingLoss(similarity, gold_similarity, args.margin, args.gold_margin, args.gold_weight)
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            loss.backward()
            if step_cnt == args.accumulate_step:
                # optimize step      
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_norm)
                step_cnt = 0
                sim_step += 1
                all_step_cnt += 1
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                s_optimizer.step()
                s_optimizer.zero_grad()
            if sim_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                print(f"similarity: {similarity[:, :10]}")
                if not args.no_gold:
                    print(f"gold similarity: {gold_similarity}")
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f"%(epoch, i / args.accumulate_step, 
                 avg_loss / args.report_freq))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.print()
                avg_loss = 0
            del similarity, gold_similarity, loss

            if all_step_cnt % args.eval_freq == 0 and all_step_cnt != 0 and step_cnt == 0:
                loss,rouge1, rouge2, rougeLsum = test(val_dataloader, scorer, args, gpuid)
                if loss < minimum_loss and is_master:
                    minimum_loss = loss
                    if is_mp:
                        recorder.save(scorer.module, "scorer.bin")
                    else:
                        recorder.save(scorer, "scorer.bin")
                    recorder.save(s_optimizer, "optimizer.bin")
                    recorder.print("best - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                if is_master:
                    recorder.print("val rouge: %.6f"%(1 - loss))
               

def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0)
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("-l", "--log", action="store_true")
    parser.add_argument("-p", "--port", type=int, default=12355)
    parser.add_argument("--log_name", default="default", type=str,
                            help="the log name to store the traing statistics")
    parser.add_argument("--state_dict_path", default="", type=str, 
                            help='load model state dict from a trained ranker')
    parser.add_argument("--model_name_or_path", default="", type=str, 
                            help='the model name in huggingface hub or the local check point')
    parser.add_argument("--cache_data", action="store_true", 
                            help='whether to cache data in memory, useful when training on cloud with blob container')
    parser.add_argument("--encode_mode", default=None, type=str)
    parser.add_argument('--train_batch_size', default =1, type=int)
    parser.add_argument('--eval_batch_size', default =8, type=int)
    parser.add_argument('--epoch', default =5, type=int)
    parser.add_argument("--report_freq", default =100, type=int,
                            help="number of steps for each report in log")
    parser.add_argument("--eval_freq", default=100, type=int,
                            help="number of steps for each evaluation")
    parser.add_argument("--accumulate_step", default =12, type=int)
    parser.add_argument("--margin",default = 0.01, type=float)
    parser.add_argument("--gold_margin", default =0, type=float)
    parser.add_argument("--warmup_steps", default =10000, type=int)
    parser.add_argument("--grad_norm", default =0, type=float)
    parser.add_argument("--seed", default =42, type=int)
    parser.add_argument("--no_gold", action="store_true")
    parser.add_argument("--max_lr", default =2e-3, type=float)
    parser.add_argument("--scale", default =1, type=float)
    parser.add_argument("--datatype", default ="diverse", type=str)
    parser.add_argument("--dataset", default ="samsum", type=str, help='dataset name')
    parser.add_argument("--max_len", default =120, type=int)  # 120 for cnndm and 80 for xsum
    parser.add_argument("--max_num", default =16, type=int)
    parser.add_argument("--num_beams", default=40, type=int, 
                            help='the beam size of used dataset, max_num should not exceed num_beams')
    parser.add_argument("--cand_weight", default =1, type=float)
    parser.add_argument("--gold_weight", default =1, type=float)
    args = parser.parse_args()

    
    # load sentence tokenizer
    nltk.download('punkt')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    setattr(args, 'sent_detector', sent_detector)

    # add assertion here
    if args.max_num > args.num_beams:
        raise ValueError(
            f"The max_num should not be larger then num_beams"
        )


    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
        elif len(args.gpuid) == 1:    
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)