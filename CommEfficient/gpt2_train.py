import os
from pprint import pformat

from pytorch_transformers import (AdamW, OpenAIGPTDoubleHeadsModel,
                                  OpenAIGPTTokenizer, GPT2DoubleHeadsModel,
                                  GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from fed_aggregator import FedOptimizer, FedModel
from utils import make_logdir, PiecewiseLinear
from utils import TableLogger, Timer, union, steps_per_epoch
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from utils import parse_args, Logger
import math

from torch.utils.data import DataLoader
from data_utils import FedSampler
from data_utils import personachat_collate_fn, FedPERSONA

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
    loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef)
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
        args, log_dir, writer, logger=None, timer=None):
    timer = timer or Timer()

    total_download = 0
    total_upload = 0
    for epoch in range(math.ceil(args.num_epochs)):
        if epoch == math.ceil(args.num_epochs) - 1:
            epoch_fraction = args.num_epochs - epoch
        else:
            epoch_fraction = 1
        mean_train_loss, download, upload = run_batches(
                model, opt, scheduler, train_loader, args, timer, epoch=epoch,
                epoch_fraction=epoch_fraction, training=True, logger=logger,
                writer=writer
            )

        # fed_aggregator's download tracking only works for the first epoch
        if epoch == 0:
            download_mb = download.sum().item() / (1024*1024)
            upload_mb = upload.sum().item() / (1024*1024)
            total_download += download_mb
            total_upload += upload_mb
    print("Total Download (MiB): {:0.2f} (only epoch 1)".format(total_download))
    print("Total Upload (MiB): {:0.2f} (only epoch 1)".format(total_upload))
    print("Avg Download Per Client: {:0.2f} (only epoch 1)".format(
        total_download / train_loader.dataset.num_clients
    ))
    print("Avg Upload Per Client: {:0.2f} (only epoch 1)".format(
        total_upload / train_loader.dataset.num_clients
    ))
    model.save_pretrained(log_dir)
    test_gpt2(model, val_loader, args, timer=timer, writer=writer)

def test_gpt2(model, val_loader, args, logger=None, timer=None, writer=None):
        nll, acc, ppl = run_batches(model, None, None, val_loader, args,
                                    timer, training=False, epoch_fraction=1,
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
        valLogger = logger or TableLogger()
        print()
        valLogger.append(epoch_stats)

def run_batches(model, opt, lr_scheduler, loader, args, timer, training,
                epoch=None, epoch_fraction=None, logger=None, writer=None):
    if not training and epoch_fraction != 1:
        raise ValueError("Must do full epochs for val")
    if epoch_fraction > 1 or epoch_fraction <= 0:
        msg = "Invalid epoch_fraction {}.".format(epoch_fraction)
        msg += " Should satisfy 0 < epoch_fraction <= 1"
        raise ValueError(msg)

    model.train(training)
    client_download = torch.zeros(loader.dataset.num_clients)
    client_upload = torch.zeros(loader.dataset.num_clients)
    spe = steps_per_epoch(args.local_batch_size,
                          loader.dataset,
                          args.num_workers)

    if training:
        epoch_idxs = epoch * spe
        losses = []
        for batch_idx, batch in enumerate(loader):
            if batch_idx > 2 and args.do_test and batch_idx < spe - 10:
                print("skipping ", batch_idx)
                continue
            # only carry out an epoch_fraction portion of the epoch
            if batch_idx > spe * epoch_fraction:
                break
            lr_scheduler.step()
            if lr_scheduler.get_lr() == 0:
                # hack to get the starting LR right for fedavg
                opt.step()

            if args.local_batch_size == -1:
                expected_num_clients = args.num_workers
                if torch.unique(batch[0]).numel() < expected_num_clients:
                    # skip if there weren't enough clients left
                    print("SKIPPING BATCH: NOT ENOUGH CLIENTS")
                    continue
            else:
                expected_numel = args.num_workers * args.local_batch_size
                if batch[0].numel() < expected_numel:
                    # skip incomplete batches
                    print("SKIPPING BATCH: NOT ENOUGH DATA")
                    continue

            loss, download, upload = model(batch)

            client_download += download
            client_upload += upload

            opt.step()
            loss = np.mean(loss)
            losses.append(loss)
            train_time = timer()
            download_mb = download.sum().item() / (1024*1024)
            upload_mb = upload.sum().item() / (1024*1024)
            batch_stats = {
                'train_time': train_time,
                'train_loss': loss,
                'total_time': timer.total_time,
                'down (MiB)': round(download_mb),
                'up (MiB)': round(upload_mb),
            }
            lr = lr_scheduler.get_lr()[0]

            writer.add_scalar('training/loss', loss, batch_idx + epoch_idxs)
            writer.add_scalar('Lr', lr, batch_idx + epoch_idxs)
            writer.add_scalar('Time/train', train_time, batch_idx + epoch_idxs)
            summary = union({'batch_idx': batch_idx+1 + epoch_idxs,
                             'lr': lr},
                            batch_stats)
            logger.append(summary)
        return np.mean(losses), client_download, client_upload

    else:
        nlls, accs, ppls = [], [], []
        for batch_idx, batch in enumerate(loader):
            if batch_idx > 5 and args.do_test and batch_idx < spe - 5:
                print("skipping ", batch_idx)
                continue
            nll, acc = model(batch)
            nll = np.mean(nll)
            acc = np.mean(acc)
            nlls.append(nll)
            accs.append(acc)
        return np.mean(nlls), np.mean(accs), np.exp(np.mean(nlls))

def train():
    args = parse_args(default_lr=4e-2)

    print(args)

    timer = Timer()
    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    if "gpt2" in args.model_checkpoint:
        tokenizer_class = GPT2Tokenizer
        model_class = GPT2DoubleHeadsModel
    else:
        tokenizer_class = OpenAIGPTTokenizer
        model_class = OpenAIGPTDoubleHeadsModel

    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    if args.do_finetune:
        if not args.do_test:
            args.model_checkpoint = args.finetune_path
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
    spe = steps_per_epoch(args.local_batch_size,
                          train_loader.dataset,
                          args.num_workers)
    print("Steps per epoch", spe)
    lr_schedule = PiecewiseLinear(
            [0, args.num_epochs * spe],
            [args.lr_scale, 0.0])
    lambda_step = lambda x: lr_schedule(x)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=[lambda_step])
    if args.do_finetune:
        test_gpt2(model, val_loader, args, logger=TableLogger(), timer=timer, writer=writer)
    else:
        train_gpt2(model, optimizer, scheduler, train_loader, val_loader, args,
               log_dir, writer=writer, logger=TableLogger(), timer=timer)
    model.finalize()

def get_data_loaders(args, tokenizer):
    dataset_class = globals()["Fed" + args.dataset_name]
    train_dataset = dataset_class(
            dataset_dir=args.dataset_dir,
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            num_candidates=args.num_candidates,
            max_history=args.max_history,
            personality_permutations=args.personality_permutations,
            do_iid=args.do_iid,
            num_clients=args.num_clients,
            train=True)
    val_dataset = dataset_class(
            dataset_dir=args.dataset_dir,
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            num_candidates=args.num_candidates,
            max_history=args.max_history,
            personality_permutations=1,
            train=False)
    train_sampler = FedSampler(train_dataset,
                               args.num_workers,
                               args.local_batch_size)
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              collate_fn=personachat_collate_fn,
                              num_workers=args.train_dataloader_workers)

    val_batch_size = args.valid_batch_size * args.num_workers
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                            collate_fn=personachat_collate_fn,
                            shuffle=False,
                            num_workers=args.val_dataloader_workers)
    """
    x = next(iter(train_loader))
    print(x)
    y = next(iter(val_loader))
    print(y)
    """

    return train_loader, val_loader

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    print("MY PID:", os.getpid())
    # reproducibility
    np.random.seed(21)
    torch.random.manual_seed(21)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train()
