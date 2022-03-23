from os.path import join
import sys

sys.path += ['../']
import argparse
import glob
import json
import logging
import os
import random
import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
#  
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from model.models import BiBertEncoder, HFBertEncoder, Reranker
from utils.lamb import Lamb
import random
from transformers import (
    AdamW,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import pandas as pd

logger = logging.getLogger(__name__)
from utils.util import (
    set_seed,
    is_first_worker,
    TraditionDataset
)
from utils.dpr_utils import (
    load_states_from_checkpoint,
    get_model_obj,
    CheckpointState,
    all_gather_list
)
import collections
retrieverBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "q_ids",
        "q_attn_mask",
        "c_ids",
        "c_attn_mask",
        "c_q_mapping",
        "is_positive",
    ],
)

def get_optimizer(args, model: nn.Module, weight_decay: float = 0.0,
                  lr=0.0, eps=0.0) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    if args.optimizer == "adamW":
        return AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    elif args.optimizer == "lamb":
        return Lamb(optimizer_grouped_parameters, lr=lr, eps=eps)
    else:
        raise Exception("optimizer {0} not recognized! Can only be lamb or adamW".format(args.optimizer))


def get_bert_reader_components(args, **kwargs):
    encoder = HFBertEncoder.init_encoder(
        args, model_type=args.reranker_model_type
    )
    hidden_size = encoder.config.hidden_size
    reranker = Reranker(encoder, hidden_size)

    return reranker


def train(args, model, reranker_model, tokenizer, global_step=0):
    """ Train the model """
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    model.to(args.device)
    reranker_model.to(args.device)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # nll loss for query

    optimizer = get_optimizer(args, model, weight_decay=args.weight_decay,
                              lr=args.learning_rate, eps=args.adam_epsilon)
    reranker_optimizer = get_optimizer(args, reranker_model, weight_decay=args.weight_decay,
                                      lr=args.reranker_learning_rate, eps=args.adam_epsilon)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        reranker_model, reranker_optimizer = amp.initialize(reranker_model, reranker_optimizer,
                                                          opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        from apex.parallel import DistributedDataParallel as DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=False,
        )
        reranker_model = torch.nn.parallel.DistributedDataParallel(
            reranker_model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=False,
        )

    tr_loss = 0.0
    tr_normal_loss = 0.0
    tr_contr_loss = 0.0
    model.zero_grad()
    model.train()
    reranker_model.zero_grad()
    reranker_model.train()
    set_seed(args)  # Added here for reproductibility
    train_flag = 1
    step = 0
    retriever_max_step = args.max_steps * (1 - args.iteration_reranker_step / args.iteration_step)
    reranker_max_step = args.max_steps * (args.iteration_reranker_step / args.iteration_step)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.1 * retriever_max_step, num_training_steps=retriever_max_step
    )
    reranker_scheduler = get_linear_schedule_with_warmup(
        reranker_optimizer, num_warmup_steps=0.1 * reranker_max_step, num_training_steps=reranker_max_step
    )
    if global_step!=0:
        train_data_path = os.path.join(args.ann_dir, 'train_ce_' + str(global_step) + '.json')
        model_path = os.path.join(args.output_dir, 'checkpoint-' + str(global_step))
        reranker_model_path = os.path.join(args.output_dir, 'checkpoint-reranker' + str(global_step))
        saved_state = load_states_from_checkpoint(model_path)
        global_step = _load_saved_state(model, optimizer, scheduler, saved_state)
        saved_state = load_states_from_checkpoint(reranker_model_path)
        global_step = _load_saved_state(reranker_model, reranker_optimizer, reranker_scheduler, saved_state)
        train_dataset = TraditionDataset(train_data_path, tokenizer, num_hard_negatives=args.number_neg,
                                        max_seq_length=args.max_seq_length,max_q_length=args.max_query_length)
    else:
        train_dataset = TraditionDataset(args.origin_data_dir, tokenizer, num_hard_negatives=args.number_neg,
                                            max_seq_length=args.max_seq_length,max_q_length=args.max_query_length)

    train_sample = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sample,
                                  collate_fn=TraditionDataset.get_collate_fn(args),
                                  batch_size=args.train_batch_size, num_workers=10)
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    train_dataloader_iter = iter(epoch_iterator)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Max steps = %d", args.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Dataset example num = %d", len(train_dataset))
    logger.info("  step each epoch = %d",
                len(train_dataset) // (args.train_batch_size *
                                       args.gradient_accumulation_steps))

    while global_step < args.max_steps:
        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            train_dataloader_iter = iter(epoch_iterator)
            batch = next(train_dataloader_iter)
            dist.barrier()

        step += 1

        batch_retriever = batch['retriever']
        inputs_retriever = {"query_ids": batch_retriever[0].long().to(args.device),
                          "attention_mask_q": batch_retriever[1].long().to(args.device),
                          "input_ids_a": batch_retriever[2].long().to(args.device),
                          "attention_mask_a": batch_retriever[3].long().to(args.device)}
        # reranker forward input_ids: T, attention_mask: T, start_positions=None, end_positions=None, answer_mask=None
        batch_reranker = tuple(t.to(args.device) for t in batch['reranker'])
        inputs_reranker = {"input_ids": batch_reranker[0].long(), "attention_mask": batch_reranker[1].long()}
        local_positive_idxs = batch_retriever[4]

        if train_flag == 0:  # 0: 训练retriever：
            eps = 1e-7
            model.train()
            reranker_model.eval()
            local_q_vector, local_ctx_vectors = model(**inputs_retriever)
            retriever_local_ctx_vectors = local_ctx_vectors.reshape(local_q_vector.size(0),
                                                                  local_ctx_vectors.size(0) // local_q_vector.size(
                                                                      0), -1)
            retriever_simila = torch.einsum("bh,bdh->bd", local_q_vector, retriever_local_ctx_vectors)
            if args.scale_simmila:
                retriever_dist_p = F.softmax(retriever_simila / (local_q_vector.size(1) ** (1 / 2)), dim=1)
            else:
                retriever_dist_p = F.softmax(retriever_simila, dim=1)
            with torch.no_grad():
                output_reranker = reranker_model(**inputs_reranker)
                binary_logits, relevance_logits, _ = output_reranker
                reranker_logits = relevance_logits / args.temperature_normal
                probs = F.softmax(reranker_logits, dim=1)
                reranker_dist_p = probs


                positive_logits = relevance_logits[:,:1]
                negtive_logits = relevance_logits[:,:]
                positive_logits_expand = positive_logits.expand(negtive_logits.size())
                reward_logits = torch.stack((positive_logits_expand,negtive_logits),-1)
                reward_prob = F.softmax(reward_logits,dim=2)
                reward = torch.log(reward_prob[:,:,0]+eps)
            
            if args.normal_term == 'kl':
                loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
                normal_loss = loss_fct(torch.log(retriever_dist_p+eps), reranker_dist_p)
            elif args.normal_term == 'cross_e':
                normal_loss = -reranker_dist_p*torch.log(retriever_dist_p+eps)
                normal_loss = normal_loss.sum()/retriever_dist_p.size(0)

            
            adv_loss = reward * torch.log(retriever_dist_p+eps)
            adv_loss = adv_loss.sum()

            loss = args.adv_lambda * adv_loss + (1-args.adv_lambda) * normal_loss
            loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            tr_normal_loss += normal_loss.item()
        if train_flag == 1:  # 1: 训练reranker：
            reranker_model.train()
            model.eval()
            output_reranker = reranker_model(**inputs_reranker)
            binary_logits, relevance_logits, _ = output_reranker

            relevance_target = torch.zeros(relevance_logits.size(0), dtype=torch.long).to(args.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            contr_loss = loss_fct(relevance_logits, relevance_target)

            loss = contr_loss + 0*binary_logits.sum()
            loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, reranker_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            tr_contr_loss += contr_loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if train_flag == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            if train_flag == 1:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(reranker_optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(reranker_model.parameters(), args.max_grad_norm)
                reranker_optimizer.step()
                reranker_scheduler.step()
                reranker_model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = {}
                loss_scalar = tr_loss / args.logging_steps
                normal_loss_scalar = tr_normal_loss / args.logging_steps
                contr_loss_scalar = tr_contr_loss / args.logging_steps
                learning_rate_scalar = scheduler.get_last_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                logs["normal_loss"] = normal_loss_scalar
                logs["contr_loss"] = contr_loss_scalar
                tr_loss = 0
                tr_normal_loss = 0
                tr_contr_loss = 0
                if is_first_worker():
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

            if global_step % args.iteration_step > args.iteration_reranker_step:
                # 训练retriever
                train_flag = 0
            elif 0 < global_step % args.iteration_step < args.iteration_reranker_step:
                # 训练reranker
                train_flag = 1
            elif global_step % args.iteration_step == 0:
                if is_first_worker():
                    _save_checkpoint(args, model, optimizer, scheduler, global_step)
                    _save_reranker_checkpoint(args, reranker_model, reranker_optimizer, reranker_scheduler, global_step)
                torch.distributed.barrier()
                train_flag = 1
                break
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                if is_first_worker():
                    _save_checkpoint(args, model, optimizer, scheduler, global_step)
                    _save_reranker_checkpoint(args, reranker_model, reranker_optimizer, reranker_scheduler, global_step)
                    # tb_writer.add_scalar("dev_nll_loss/dev_avg_rank", validate_rank, global_step)
            if global_step >= args.max_steps:
                break
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()
    return global_step


def sum_main(x, opt):
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def _save_checkpoint(args, model, optimizer, scheduler, step: int) -> str:
    offset = step
    epoch = 0
    model_to_save = get_model_obj(model)
    cp = os.path.join(args.output_dir, 'checkpoint-' + str(offset))

    meta_params = {}

    state = CheckpointState(model_to_save.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                            offset,
                            epoch, meta_params
                            )
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp


def _save_reranker_checkpoint(args, model, optimizer, scheduler, step: int) -> str:
    offset = step
    epoch = 0
    model_to_save = get_model_obj(model)
    cp = os.path.join(args.output_dir, 'checkpoint-reranker' + str(offset))

    meta_params = {}

    state = CheckpointState(model_to_save.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                            offset,
                            epoch, meta_params
                            )
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp

def _load_saved_state(model, optimizer, scheduler, saved_state: CheckpointState):
    epoch = saved_state.epoch
    step = saved_state.offset
    logger.info('Loading checkpoint @ step=%s', step)

    model_to_load = get_model_obj(model)
    logger.info('Loading saved model state ...')
    model_to_load.load_state_dict(saved_state.model_dict)  # set strict=False if you use extra projection
    optimizer.load_state_dict(saved_state.optimizer_dict)
    scheduler.load_state_dict(saved_state.scheduler_dict)
    return step


def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list:",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path_ict",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--num_epoch",
        default=0,
        type=int,
        help="Number of epoch to train, if specified will use training data instead of ann",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--corpus_path",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=32,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--triplet", default=False, action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )

    parser.add_argument(
        "--optimizer",
        default="adamW",
        type=str,
        help="Optimizer - lamb or adamW",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=300000,
        type=int,
        help="If > 0: set total number of training steps to perform",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--contr_loss",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--normal_loss",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--origin_data_dir",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--origin_data_dir_dev",
        default=None,
        type=str,
    )
    # ----------------- ANN HyperParam ------------------

    parser.add_argument(
        "--load_optimizer_scheduler",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--single_warmup",
        default=True,
        action="store_true",
        help="use single or re-warmup",
    )

    parser.add_argument("--adv_data_path",
                        type=str,
                        default=None,
                        help="adv_data_path", )

    parser.add_argument("--ann_data_path",
                        type=str,
                        default=None,
                        help="adv_data_path", )
    parser.add_argument(
        "--fix_embedding",
        default=False,
        action="store_true",
        help="use single or re-warmup",
    )
    parser.add_argument(
        "--continue_train",
        default=False,
        action="store_true",
        help="use single or re-warmup",
    )
    parser.add_argument(
        "--adv_loss_alpha",
        default=0.3,
        type=float,
        help="use single or re-warmup",
    )

    parser.add_argument("--reranker_model_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--reranker_model_type", type=str, default="", help="For distant debugging.")
    parser.add_argument("--number_neg", type=int, default=20, help="For distant debugging.")
    parser.add_argument("--adv_lambda", default=0., type=float)
    parser.add_argument("--adv_steps", default=3, type=int)
    # ----------------- End of Doc Ranking HyperParam ------------------
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--test_qa_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--train_qa_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--dev_qa_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--passage_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--iteration_step", default=80, type=int)
    parser.add_argument("--iteration_reranker_step", default=40, type=int)
    parser.add_argument("--temperature_normal", default=1, type=float)

    parser.add_argument("--scale_simmila",  default=False,  action="store_true")
    parser.add_argument("--reranker_learning_rate", default=0,type=float)
    parser.add_argument("--load_cache", default=False, action="store_true")
    parser.add_argument("--ann_dir", type=str, default="", help="For distant debugging.")
    parser.add_argument("--normal_term", type=str, default="cross_e", help="For distant debugging.")
    
    parser.add_argument("--global_step", type=int, default=0, help="For distant debugging.")
    args = parser.parse_args()

    return args


def set_env(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)


def load_states_from_checkpoint_ict(model_file: str) -> CheckpointState:
    from torch.serialization import default_restore_location
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    new_stae_dict = {}
    for key, value in state_dict['model']['query_model']['language_model'].items():
        new_stae_dict['question_model.' + key] = value
    for key, value in state_dict['model']['context_model']['language_model'].items():
        new_stae_dict['ctx_model.' + key] = value
    return new_stae_dict


def load_model(args):
    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True)
    model = BiBertEncoder(args)
    if args.model_name_or_path_ict is not None:
        saved_state = load_states_from_checkpoint_ict(args.model_name_or_path_ict)
        model.load_state_dict(saved_state)
    if args.model_name_or_path is not None:
        saved_state = load_states_from_checkpoint(args.model_name_or_path)
        model.load_state_dict(saved_state.model_dict, strict=False)

    reranker_model = get_bert_reader_components(args)
    if args.reranker_model_path != '':
        reranker_saved_state = load_states_from_checkpoint(args.reranker_model_path)
        reranker_model.load_state_dict(reranker_saved_state.model_dict, strict=False)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    return tokenizer, model, reranker_model

def main():
    args = get_arguments()
    set_env(args)
    tokenizer, model,reranker_model = load_model(args)

    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    log_path = os.path.join(args.output_dir, 'log.txt')
    # sh = logging.StreamHandler()
    if  args.global_step ==0:
        handler = logging.FileHandler(log_path, 'w', 'utf-8')
    else:
        handler = logging.FileHandler(log_path, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(sh)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print(logger)
    dist.barrier()
    global_step = args.global_step
    # eval_first(args,model,0,renew_tools)
    if global_step >= args.max_steps:
        pass
    else:
        global_step = train(args, model,reranker_model, tokenizer,global_step=global_step) # 训练，然后当需要弹出的时候弹出
        logger.info(" global_step = %s", global_step)

    if args.local_rank != -1:
        dist.barrier()


if __name__ == "__main__":
    main()
