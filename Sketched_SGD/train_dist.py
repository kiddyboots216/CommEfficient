import os
import collections
import logging
import glob
import re
from functools import singledispatch
from collections import OrderedDict
from collections import namedtuple
from inspect import signature

import torch, torchvision
import numpy as np
import ray

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset

import itertools as it
import copy

from minimal import Net, cifar10, Correct, union, \
        Timer, TableLogger, normalise, pad, transpose, \
        Crop, FlipLR, Cutout, Transform, Batches, TSVLogger

from sketched_classes import SketchedModel, SketchedWorker, SketchedOptimizer, SketchedLoss

DATA_PATH = 'sample_data'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

def train(model, opt, scheduler, criterion, accuracy, 
    train_loader, val_loader, epochs, loggers=(), timer=None):
    timer = timer or Timer()
    scheduler.step()
    for epoch in range(args.epochs):
        scheduler.step()
        train_losses, train_accs = run_batches(model, opt, 
            criterion, accuracy, train_loader, True)
        train_time = timer()
        val_losses, val_accs = run_batches(model, None, 
            criterion, accuracy, val_loader, False)
        val_time = timer()
        epoch_stats = {
            'train_time': train_time,
            'train_loss': np.mean(train_losses),
            'train_acc': np.mean(train_accs),
            'test_time': val_time,
            'test_loss': np.mean(val_losses),
            'test_acc': np.mean(val_accs),
            'total_time': timer.total_time,
        }
        summary = union({'epoch': epoch+1, 'lr': scheduler.get_lr()[0]}, epoch_stats)
        for logger in loggers:
            logger.append(summary)
    return summary

def run_batches(model, opt, criterion, accuracy, loader, training):
    losses = []
    accs = []
    for idx, batch in enumerate(loader):
        inputs = batch["input"]
        targets = batch["target"]
        outs = model(inputs)
        batch_loss = criterion(outs, targets)
        if training:
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
        losses.append(batch_loss.mean())
        batch_acc = accuracy(torch.cat(ray.get(outs), dim=0), targets).float().mean().cpu().numpy()
        accs.append(batch_acc)
    return losses, accs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sketched", action="store_true")
parser.add_argument("--sketch_biases", action="store_true")
parser.add_argument("--sketch_params_larger_than", action="store_true")
parser.add_argument("-k", type=int, default=50000)
parser.add_argument("--p2", type=int, default=1)
parser.add_argument("--cols", type=int, default=500000)
parser.add_argument("--rows", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_blocks", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=24)
parser.add_argument("--epochs_per_iter", type=int, default=1)
parser.add_argument("--optimizer", type=str, default="SGD")
parser.add_argument("--criterion", type=str, default="CrossEntropyLoss")
parser.add_argument("--iterations", type=int, default=1)
parser.add_argument("--test", type=bool, default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Downloading datasets')
DATA_DIR = "sample_data"
dataset = cifar10(DATA_DIR)

train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
print('Starting timer')
timer = Timer()

print('Preprocessing training data')
train_set = list(zip(
        transpose(normalise(pad(dataset['train']['data'], 4))),
        dataset['train']['labels']))
print('Finished in {:.2f} seconds'.format(timer()))
print('Preprocessing test data')
test_set = list(zip(transpose(normalise(dataset['test']['data'])),
                    dataset['test']['labels']))
print('Finished in {:.2f} seconds'.format(timer()))

train_loader = Batches(Transform(train_set, train_transforms),
                        args.batch_size, shuffle=True,
                        set_random_choices=True, drop_last=True)
val_loader = Batches(test_set, args.batch_size, shuffle=False,
                       drop_last=False)

sketched_params = {
    "k": args.k,
    "p2": args.p2,
    "num_cols": args.cols,
    "num_rows": args.rows,
    "num_blocks": args.num_blocks,
    "lr": 1,
    "num_workers": args.num_workers,
    "momentum": args.momentum,
    "optimizer" : args.optimizer,
    "criterion": args.criterion,
    "weight_decay": 5e-4*args.batch_size/args.num_workers,
    "nesterov": args.nesterov,
    "dampening": 0,
    "batch_size": args.batch_size,
}
ray.init(num_gpus=7, redis_password="sketched_sgd")
model_cls = Net
model_config = {}
if args.test:
    model_config = {
        'channels': {'prep': 1, 'layer1': 1, 'layer2': 1, 'layer3': 1},
    }
workers = [SketchedWorker.remote(sketched_params) for _ in range(args.num_workers)]
sketched_model = SketchedModel(model_cls, model_config, workers)
opt = optim.SGD(sketched_model.parameters(), lr=1)
sketched_opt = SketchedOptimizer(opt, workers)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
sketched_criterion = SketchedLoss(criterion, workers)
accuracy = Correct().to(device)
lambda_step = lambda t: np.interp([t], [0, 5, args.epochs+1], [0, 0.4, 0])[0]
scheduler = optim.lr_scheduler.LambdaLR(sketched_opt, lr_lambda=[lambda_step])
train(sketched_model, sketched_opt, scheduler, sketched_criterion, accuracy,
    train_loader, val_loader, args.epochs, loggers=(TableLogger(), TSVLogger()), timer=timer)
