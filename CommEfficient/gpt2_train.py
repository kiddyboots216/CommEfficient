import os
import logging
from pprint import pformat

from pytorch_transformers import (AdamW, OpenAIGPTDoubleHeadsModel,
                                  OpenAIGPTTokenizer, GPT2DoubleHeadsModel,
                                  GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from gpt2_dataloader import get_data_loaders

from CommEfficient.functions import FedCommEffOptimizer, FedCommEffCriterion, FedCommEffModel, FedCommEffMetric
from utils import make_logdir
from CommEfficient.minimal import PiecewiseLinear, TableLogger, Timer, union
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from utils import parse_args

import numpy as np
import multiprocessing

logger = logging.getLogger(__file__)

ATTR_TO_SPECIAL_TOKEN = {
                         'bos_token': '<bos>',
                         'eos_token': '<eos>',
                         'pad_token': '<pad>',
                         'additional_special_tokens': ('<speaker1>',
                                                       '<speaker2>')
                        }

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model

    if they have not already been added.
    """
    orig_num_tokens = len(tokenizer.encoder)
    # returns 0 and doesn't add if they are already there
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    if num_added_tokens > 0:
        model.resize_token_embeddings(
                new_num_tokens=orig_num_tokens + num_added_tokens
            )


def train_gpt2(model, opt, scheduler, train_loader, val_loader,
               args, log_dir, logger=None, timer=None, writer=None):
    timer = timer or Timer()
    epochs = args.num_epochs
    for epoch in range(epochs):
        train_loss = run_batches(model, opt, scheduler, train_loader,
                                 args, timer, training=True,
                                 logger=logger, writer=writer)
        train_time = timer()
        model.save_pretrained(log_dir)
        nll, acc, ppl = run_batches(model, None, None, val_loader, args,
                                    timer, training=False,
                                    logger=TableLogger(), writer=writer)
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
                         'lr': lr},
                        epoch_stats)
        logger.append(summary)
        return summary

def run_batches(model, opt, scheduler, loader, args,
                timer, training, logger=None, writer=None):
    num_clients = args.num_clients
    clients = np.arange(num_clients)

    if training:
        model.train(True)
        losses = []
        for batch_idx, batch in enumerate(loader):
            loss = model(batch)
            scheduler.step()
            opt.step()
            loss = np.mean(loss)
            losses.append(loss)
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
                             'lr': lr},
                            batch_stats)
            logger.append(summary)
        return np.mean(losses)

    else:
        model.train(False)
        nlls, accs, ppls = [], [], []
        for batch_idx, batch in enumerate(loader):
            nll, acc = model(batch)
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

def train():
    args = parse_args(default_lr=4e-2)

    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    timer = Timer()
    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    if "gpt2" in args.model_checkpoint:
        tokenizer_class = GPT2Tokenizer
        model_class = GPT2DoubleHeadsModel
    else:
        tokenizer_class = OpenAIGPTTokenizer
        model_class = OpenAIGPTDoubleHeadsModel

    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)

    args.len_tokenizer = len(tokenizer)

    # Do logging now before we overwrite model
    log_dir = make_logdir(args)
    writer = SummaryWriter(log_dir=log_dir)
    tokenizer.save_pretrained(log_dir)
    getattr(model, 'module', model).config.to_json_file(
            os.path.join(log_dir, CONFIG_NAME)
        )
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    # HAVE TO USE SGD FOR FED
    optimizer = SGD(model.parameters(), lr=1)

    logger.info('Finished in {:.2f} seconds'.format(timer()))
    logger.info("Prepare datasets")
    loaders = get_data_loaders(args, tokenizer)
    train_loader, val_loader = loaders

    logger.info('Finished in {:.2f} seconds'.format(timer()))
    logger.info("Initializing everything")
    model = FedCommEffModel(model, args)
    optimizer = FedCommEffOptimizer(optimizer, args)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    metric = FedCommEffMetric(criterion, args)
    criterion = FedCommEffCriterion(criterion, args)
    lr_schedule = PiecewiseLinear(
            [0, args.num_epochs * len(train_loader)],
            [args.lr_scale, 0.0])
    lambda_step = lambda x: lr_schedule(x)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=[lambda_step])
    train_gpt2(model, optimizer, scheduler, train_loader, val_loader, args,
               log_dir, logger=TableLogger(), timer=timer, writer=writer)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    train()
