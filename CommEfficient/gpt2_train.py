# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
from datetime import datetime
import os
import math
import logging
import json
import tarfile
import tempfile
import socket
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from pytorch_transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)
from pytorch_transformers import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

logger = logging.getLogger(__file__)

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()

    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir


def get_dataset(tokenizer, dataset_path, dataset_cache=None):
    """ Get PERSONACHAT from S3 """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)
    return dataset

def get_dataset_personalities(tokenizer, dataset_path, dataset_cache=None):
    """ Get personalities from PERSONACHAT """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        personachat = torch.load(dataset_cache)
    else:
        logger.info("Download PERSONACHAT dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            personachat = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        personachat = tokenize(personachat)
        torch.save(personachat, dataset_cache)

    logger.info("Filter personalities")
    personalities = []
    for dataset in personachat.values():
        for dialog in dataset:
            personalities.append(dialog["personality"])

    logger.info("Gathered {} personalities".format(len(personalities)))
    return personalities

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# FedSketched imports
from CommEfficient.functions import FedCommEffOptimizer, FedCommEffCriterion, make_logdir, FedCommEffModel, FedCommEffMetric
from CommEfficient.minimal import PiecewiseLinear, TableLogger, TSVLogger, Timer, union
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import multiprocessing

global start_idx
start_idx = 0

def train_gpt2(model, opt, scheduler, train_loader, val_loader, params, log_dir, logger=None, timer=None, writer=None):
    timer = timer or Timer()
    epochs = params["epochs"]
    for epoch in range(epochs):
        train_loss = run_batches(model, opt, scheduler, train_loader, params, timer, training=True, logger=logger, writer=writer)
        train_time = timer()
        model.save_pretrained(log_dir)
        nll, acc, ppl = run_batches(model, None, None, val_loader, params, timer, training=False, logger=TableLogger(), writer=writer)
        val_time = timer()
        epoch_stats = {
            'train_time': train_time,
            'train_loss': train_loss,
            'val_nll': nll,
            'val_acc': acc,
            'val_ppl': ppl,
            'val_time': val_time,
            'total_time': timer.total_time,
        }
        writer.add_scalar('validation/nll', nll)
        writer.add_scalar('validation/acc', acc)
        writer.add_scalar('validation/ppl', ppl)
        logger = TableLogger()
        lr = scheduler.get_lr()[0]
        summary = union({'epoch': epoch+1,
            'lr': lr}, epoch_stats)
        logger.append(summary)
        return summary

def run_batches(model, opt, scheduler, loader, params, timer, training, logger=None, writer=None):
    participation = params['participation']
    n_clients = params['n_clients']
    device = params['device']
    batch_size = params['train_batch_size']
    clients = np.arange(n_clients)
    n_clients_to_select = int(n_clients * participation)
    model.train(training)

    if training:
        losses = []
        global start_idx
        for batch_idx, batch in enumerate(loader):
            #print(f"Batch size: {[b.size() for b in batch]}")
            start_idx = start_idx % n_clients
            end_idx = start_idx + n_clients_to_select
            idx = np.random.choice(clients, 
                n_clients_to_select, replace=False)
            #print(f"Selecting randomly {idx}")
            idx = np.arange(start_idx, end_idx)
            #print(f"Selecting in order {idx}")
            minibatches = []
            for i, _ in enumerate(idx):
                start = i * batch_size 
                end = (i+1) * batch_size 
                minibatch = [b[start:end] for b in batch]
                minibatches.append(minibatch)
            loss = model(minibatches, idx)
            scheduler.step()
            opt.step(idx)
            loss = np.mean(loss)
            losses.append(loss)
            start_idx = end_idx
            train_time = timer()
            batch_stats = {
                'train_time': train_time,
                'train_loss': loss,
                'total_time': timer.total_time,
            }
            lr = scheduler.get_lr()[0]

            writer.add_scalar('training/loss', loss, batch_idx)
            writer.add_scalar('Lr', lr, batch_idx)
            writer.add_scalar('Time/train', train_time, batch_idx)
            summary = union({'batch_idx': batch_idx+1,
                'lr': lr}, batch_stats)
            logger.append(summary)
            """
            if params['test']:
                break
            """
        return np.mean(losses)

    else:
        nlls, accs, ppls = [], [], []
        for batch_idx, batch in enumerate(loader):
            idx = np.arange(n_clients_to_select)
            minibatches = []
            batch_len = batch[3].size()[0]
            indices = []
            batch_size = params["valid_batch_size"] 
            for i, _ in enumerate(idx):
                start = i * batch_size // n_clients_to_select
                end = (i+1) * batch_size // n_clients_to_select
                if end > batch_len:
                    break
                minibatch = [b[start:end] for b in batch]
                minibatches.append(minibatch)
                indices.append(i)
            nll, acc = model(minibatches, indices)
            """
            nll, acc, ppl = model(minibatches, indices)
            ppl = np.mean(ppl)
            ppls.append(ppl)
            """
            nll = np.mean(nll)
            acc = np.mean(acc)
            nlls.append(nll)
            accs.append(acc)
            """
            val_time = timer()
            batch_stats = {
                'test_time': val_time,
                'test_nll': nll,
                'test_acc': acc,
                'test_ppl': ppl,
                'total_time': timer.total_time,
            }
            summary = union({'batch_idx': batch_idx+1, }, batch_stats)
            logger.append(summary)
            """
        return np.mean(nlls), np.mean(accs), np.exp(np.mean(ppls))


SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ('<speaker1>', '<speaker2>')}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN)  # returns 0 and doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(
            new_num_tokens=orig_num_tokens + num_added_tokens)

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    instance = {}
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance, sequence  # TODO: second arg is never used, delete it


def get_data_loaders(args, tokenizer, test=False):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    dialogs_processed = 0
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            dialogs_processed += 1
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance, _ = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                persona = [persona[-1]] + persona[:-1]  # permuted personalities
            if test and dialogs_processed > args.n_dialogs:
                break

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = None
    valid_sampler = None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(True))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=4e-2, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")

    # FedSketched args
    parser.add_argument("--k", type=int, default=50000)
    parser.add_argument("--p2", type=int, default=4)
    parser.add_argument("--cols", type=int, default=500000)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--num_blocks", type=int, default=20)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--test", action="store_true")
    momentum_types = ["none", "local", "virtual"]
    parser.add_argument("--momentum_type", choices=momentum_types,
                        default="none")
    error_types = momentum_types
    parser.add_argument("--error_type", choices=error_types,
                        default="none")
    parser.add_argument("--topk_down", action="store_true")
    modes = ["sketch", "true_topk", "local_topk"]
    parser.add_argument("--mode", choices=modes, default="sketch")
    reductions = ["sum", "mean", "median"]
    parser.add_argument("--grad_reduce", choices=reductions, default="sum")
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--participation", type=float, default=1.0)
    parser.add_argument("--n_dialogs", type=int, default=1)
    args = parser.parse_args()
    if args.test:
        args.train_batch_size = 2
        args.gradient_accumulation_steps = 2
        args.valid_batch_size = 2
        args.epochs = 1
        args.participation = 1.0

    args.workers = int(args.clients * args.participation)

    params = {
        "device": args.device,
        "test": args.test,
        "model": "gpt2",
        # sketching params
        "k": args.k,
        "p2": args.p2,
        "num_cols": args.cols,
        "num_rows": args.rows,
        "batch_size": args.train_batch_size,
        "num_blocks": args.num_blocks,
        # federation params
        "n_clients": args.clients,
        "participation": args.participation, 
        "n_workers": args.workers,
        # optimizer params
        "lr": 1, #assumes scheduler accounts for this lr
        "momentum": args.momentum,
        "nesterov": args.nesterov,
        "dampening": 0,
        "epochs": args.n_epochs,
        "grad_reduce": args.grad_reduce,
        "weight_decay": 0,
        # algorithmic params
        "mode": args.mode,
        "topk_down": args.topk_down,
        "momentum_type": args.momentum_type,
        "error_type": args.error_type,
        # GPT2 SPECIFIC
        "model_checkpoint": args.model_checkpoint,
        "max_norm": args.max_norm,
        "lm_coef": args.lm_coef,
        "mc_coef": args.mc_coef,
        "grad_accum_steps": args.gradient_accumulation_steps,
        "mean_grads": True,
        # model outs
        "n_results_train": 1,
        "n_results_val": 2,
    }
    args.train_batch_size *= params["grad_accum_steps"]
    params["train_batch_size"] = args.train_batch_size
    args.train_batch_size *= params["n_workers"]
    args.valid_batch_size *= params["n_workers"]
    params["valid_batch_size"] = args.valid_batch_size

    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    timer = Timer()
    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    # SKETCHED
    params["len_tokenizer"] = len(tokenizer)

    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    # Do logging now before we overwrite model
    log_dir = make_logdir(params)
    writer = SummaryWriter(log_dir=log_dir)
    tokenizer.save_pretrained(log_dir)
    getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    # HAVE TO USE SGD FOR FED
    optimizer = SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"])

    logger.info('Finished in {:.2f} seconds'.format(timer()))
    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer, params['test'])

    logger.info('Finished in {:.2f} seconds'.format(timer()))
    logger.info("Initializing everything")
    model = FedCommEffModel(model, params)
    optimizer = FedCommEffOptimizer(optimizer, params)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion = FedCommEffCriterion(criterion, params)
    metric = FedCommEffMetric(None, params)
    lr_schedule = PiecewiseLinear(
            [0, args.n_epochs * len(train_loader)], 
            [args.lr, 0.0])
    lambda_step = lambda x: lr_schedule(x)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_step])
    train_gpt2(model, optimizer, scheduler, train_loader, val_loader, params, log_dir, logger=TableLogger(), timer=timer, writer=writer)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    train()
