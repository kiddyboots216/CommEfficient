import os
from pprint import pformat

from pytorch_transformers import (AdamW, OpenAIGPTDoubleHeadsModel,
                                  OpenAIGPTTokenizer, GPT2DoubleHeadsModel,
                                  GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from fed_aggregator import FedOptimizer, FedModel
from utils import make_logdir
from utils import TableLogger, Timer, union
from models.configs import PiecewiseLinear
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from utils import parse_args, Logger

from torch.utils.data import DataLoader
from data_utils import FedSampler
from data_utils import personachat_collate_fn, FedPersonaChat

import numpy as np
import torch.multiprocessing as multiprocessing

logger = Logger()

ATTR_TO_SPECIAL_TOKEN = {
                         'bos_token': '<bos>',
                         'eos_token': '<eos>',
                         'pad_token': '<pad>',
                         'additional_special_tokens': ('<speaker1>',
                                                       '<speaker2>')
                        }

def _check_shape(y_pred, y):
    if y.ndimension() > 1 and y.shape[1] == 1:
        # (N, 1, ...) -> (N, ...)
        y = y.squeeze(dim=1)
    if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
        # (N, 1, ...) -> (N, ...)
        y_pred = y_pred.squeeze(dim=1)
    if not (y.ndimension() == y_pred.ndimension()
            or y.ndimension() + 1 == y_pred.ndimension()):
        raise ValueError("y must have shape of (batch_size, ...) and "
                         "y_pred must have shape of (batch_size, "
                         "num_categories, ...) or (batch_size, ...), but "
                         "given {} vs {}.".format(y.shape, y_pred.shape))
    y_shape = y.shape
    y_pred_shape = y_pred.shape
    if y.ndimension() + 1 == y_pred.ndimension():
        y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]
    if not (y_shape == y_pred_shape):
        raise ValueError("y and y_pred must have compatible shapes.")
    return y_pred, y

def inference(model, batch, args):
    model.eval()
    with torch.no_grad():
        (input_ids, mc_token_ids, lm_labels,
                mc_labels, token_type_ids) = batch
        lm_logits, mc_logits, *_ = model(input_ids,
                                         token_type_ids=token_type_ids,
                                         mc_token_ids=mc_token_ids)
        lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(
                -1, lm_logits.size(-1)
            )
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
        return ((lm_logits_flat_shifted, mc_logits),
                (lm_labels_flat_shifted, mc_labels))

def accuracy(y_pred, y):
    y_pred, y = _check_shape(y_pred, y)
    indices = torch.argmax(y_pred, dim=1)
    correct = torch.eq(indices, y).view(-1)
    return torch.sum(correct).float() / correct.shape[0]

nll_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
def compute_loss_val(model, batch, args):
    (input_ids, mc_token_ids, lm_labels,
            mc_labels, token_type_ids) = batch

    logits, labels = inference(model, batch, args)
    lm_logits, mc_logits = logits
    lm_labels, mc_labels = labels
    nll = nll_criterion(lm_logits, lm_labels)
    acc = accuracy(mc_logits, mc_labels)
    return nll, acc

def compute_loss_train(model, batch, args):
    (input_ids, mc_token_ids, lm_labels,
            mc_labels, token_type_ids) = batch

    lm_loss, mc_loss, *_ = model(
        input_ids, token_type_ids=token_type_ids,
        mc_token_ids=mc_token_ids,
        mc_labels=mc_labels, lm_labels=lm_labels
    )
    loss = ((lm_loss * args.lm_coef + mc_loss * args.mc_coef)
            / args.num_train_batch_shards)
    # there are no metrics, but still need to return a tuple
    return loss,

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
        mean_train_loss = run_batches(model, opt, scheduler, train_loader,
                                 args, timer, training=True,
                                 logger=logger, writer=writer)
        model.save_pretrained(log_dir)
        nll, acc, ppl = run_batches(model, None, None, val_loader, args,
                                    timer, training=False,
                                    logger=TableLogger(), writer=writer)
        val_time = timer()
        epoch_stats = {
            #'mean_train_loss': mean_train_loss,
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
            loss, = model(batch)
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
            if batch_idx > 10 and args.do_test:
                break
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
    model = FedModel(model, compute_loss_train, args, compute_loss_val)
    optimizer = FedOptimizer(optimizer, args)
    lr_schedule = PiecewiseLinear(
            [0, args.num_epochs * len(train_loader)],
            [args.lr_scale, 0.0])
    lambda_step = lambda x: lr_schedule(x)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=[lambda_step])
    train_gpt2(model, optimizer, scheduler, train_loader, val_loader, args,
               log_dir, logger=TableLogger(), timer=timer, writer=writer)
    model.finalize()

def get_data_loaders(args, tokenizer):
    train_dataset = FedPersonaChat(args.dataset_dir,
                                   tokenizer,
                                   args.num_candidates,
                                   args.max_history,
                                   do_iid=args.do_iid,
                                   num_clients=args.num_clients,
                                   train=True)
    val_dataset = FedPersonaChat(args.dataset_dir,
                                 tokenizer,
                                 args.num_candidates,
                                 args.max_history,
                                 train=False)
    train_sampler = FedSampler(train_dataset,
                               args.num_workers,
                               args.local_batch_size)
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              collate_fn=personachat_collate_fn,
                              num_workers=4)

    val_batch_size = args.local_batch_size * args.num_workers
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                            collate_fn=personachat_collate_fn,
                            shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    train()
