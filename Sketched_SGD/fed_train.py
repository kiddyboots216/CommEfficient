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

from minimal import Net, cifar10, Correct, union, PiecewiseLinear, \
        Timer, TableLogger, normalise, pad, transpose, \
        Crop, FlipLR, Cutout, Transform, Batches, TSVLogger

from sketched_classes import *
from fed_sketched_classes import *
from fed_param_server import *

DATA_PATH = 'sample_data'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

def train(model, opt, scheduler, criterion, 
    accuracy, train_loader, val_loader, 
    params, loggers=(), timer=None, fed_params=None):
    timer = timer or Timer()
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    for epoch in range(args.epochs):
        train_loss, train_acc = run_batches(model, opt, scheduler, 
            criterion, accuracy, train_loader, True)
        train_time = timer()
        val_loss, val_acc = run_batches(model, None, scheduler,
            criterion, accuracy, val_loader, False)
        val_time = timer()
        epoch_stats = {
            'train_time': train_time,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_time': val_time,
            'test_loss': val_loss,
            'test_acc': val_acc,
            'total_time': timer.total_time,
        }
        summary = union({'epoch': epoch+1, 
            'lr': scheduler.get_lr()[0]*batch_size}, epoch_stats)
        for logger in loggers:
            logger.append(summary)
    return summary

def run_batches(model, opt, scheduler, criterion, 
    accuracy, loader, training, fed_params):
    losses = []
    accs = []
    for idx, batch in enumerate(loader):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        outs = model(inputs)
        batch_loss = criterion(outs, targets)
        if training:
            opt.zero_grad()
            #batch_loss.sum().backward()
            batch_loss.backward()
            scheduler.step()
            opt.step()
        #losses.append(batch_loss.detach().mean().cpu().numpy())
        losses.append(batch_loss.mean())
        #batch_acc = accuracy(outs, targets).float().mean().cpu().numpy()
        batch_acc = accuracy(torch.cat(ray.get(outs), dim=0), 
                targets).float().mean().cpu().numpy()
        accs.append(batch_acc)
    return np.mean(losses), np.mean(accs)

def train_fed(model, opt, scheduler, criterion, 
    accuracy, train_loader, val_loader, 
    params, loggers=(), timer=None, fed_params=None):
    timer = timer or Timer()
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    for epoch in range(args.epochs):
        train_loss, train_acc = run_fed_batches(model, opt, scheduler, 
            criterion, accuracy, train_loader, True, params)
        train_time = timer()
        val_loss, val_acc = run_fed_batches(model, None, scheduler,
            criterion, accuracy, val_loader, False, params)
        val_time = timer()
        epoch_stats = {
            'train_time': train_time,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_time': val_time,
            'test_loss': val_loss,
            'test_acc': val_acc,
            'total_time': timer.total_time,
        }
        summary = union({'epoch': epoch+1, 
            'lr': scheduler.get_lr()[0]*batch_size}, epoch_stats)
        for logger in loggers:
            logger.append(summary)
    return summary

def run_fed_batches(model, opt, scheduler, criterion, 
    accuracy, loaders, training, fed_params):
    participation = fed_params['participation_rate']
    model.train(training)
    losses = []
    accs = []
    if training:
        for _ in range(int(1/participation)):
            outs = model(loaders)
            batch_loss = criterion(outs)
            print(f"Loss: {batch_loss.mean()}")
            opt.zero_grad()
            #batch_loss.sum().backward()
            batch_loss.backward()
            scheduler.step()
            opt.step()
            #losses.append(batch_loss.detach().mean().cpu().numpy())
            losses.append(batch_loss.mean())
            # TODO: Fix train acc calculation
            #batch_acc = accuracy(outs, targets).float().mean().cpu().numpy()
            #batch_acc = accuracy(torch.cat(ray.get(outs), dim=0), 
            #        targets).float().mean().cpu().numpy()
            #accs.append(batch_acc)
    else:
        for idx, batch in enumerate(loaders):
            inputs, targets = batch
            outs = model(inputs, targets)
            batch_loss = criterion(outs, targets)
            losses.append(batch_loss.mean())
            batch_acc = accuracy(ray.get(outs), 
                    targets.cuda()).float().mean().cpu().numpy()
            accs.append(batch_acc)
    return np.mean(losses), np.mean(accs)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-k", type=int, default=50000)
parser.add_argument("-p2", type=int, default=1)
parser.add_argument("-cols", type=int, default=500000)
parser.add_argument("-rows", type=int, default=5)
parser.add_argument("-num_workers", type=int, default=1)
parser.add_argument("-num_blocks", type=int, default=1)
parser.add_argument("-batch_size", type=int, default=512)
parser.add_argument("-nesterov", action="store_true")
parser.add_argument("-momentum", type=float, default=0.9)
parser.add_argument("-epochs", type=int, default=24)
parser.add_argument("-test", action="store_true")
parser.add_argument("-fed", action="store_true")
parser.add_argument("-clients", type=int, default=1)
parser.add_argument("-rate", type=float, default=1.0)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timer = Timer()
if args.fed:
    from data_utils import *
    hp_default["participation_rate"] = args.rate
    hp_default["n_clients"] = args.clients
    DATA_LEN = 50000
    #hp_default["batch_size"] = int(DATA_LEN/hp_default["n_clients"])
    #hp_default["batch_size"] = args.batch_size
    hp_default["batch_size"] = int(args.batch_size/(args.rate * args.clients))
    train_loader, central_train_loader, val_loader, stats = get_data_loaders(hp_default, verbose=True)
    fed_params = hp_default
    fed_params["batch_size"] = args.batch_size
    fed_params["epochs"] = args.epochs
else:
    print('Downloading datasets')
    DATA_DIR = "sample_data"
    dataset = cifar10(DATA_DIR)

    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
    print('Starting timer')

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

print('Initializing everything')
sketched_params = {
    "k": args.k,
    "p2": args.p2,
    "num_cols": args.cols,
    "num_rows": args.rows,
    "num_blocks": args.num_blocks,
    "lr": 1,
    "num_workers": args.num_workers,
    "momentum": args.momentum,
    "weight_decay": 5e-4*args.batch_size,
    "nesterov": args.nesterov,
    "dampening": 0,
    "batch_size": args.batch_size,
    "epochs": args.epochs,
}
ray.init(num_gpus=7, redis_password="sketched_sgd")
model_cls = Net
model_config = {}
if args.test:
    model_config = {
        'channels': {'prep': 1, 'layer1': 1, 
        'layer2': 1, 'layer3': 1},
    }
else:
    model_config = {
            'channels': {'prep': 64, 'layer1': 128, 
            'layer2': 256, 'layer3': 512},
    }
lr_schedule = PiecewiseLinear([0, 5, args.epochs], [0, 0.4, 0])
lambda_step = lambda step: lr_schedule(step/len(train_loader))/args.batch_size
if args.fed:
    workers = [FedSketchedWorker.remote(sketched_params) for _ in range(args.num_workers)]
    param_server = FedParamServer.remote(sketched_params)
    model = FedSketchedModel(model_cls, model_config, workers, param_server, fed_params)
    opt = optim.SGD(model.parameters(), lr=1)
    opt = FedSketchedOptimizer(opt, workers, param_server, model)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion = FedSketchedLoss(criterion, workers, param_server, model)
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda_step])
    accuracy = Correct().to(device)
else:
    workers = [SketchedWorker.remote(sketched_params) for _ in range(args.num_workers)]
    model = SketchedModel(model_cls, model_config, workers)
    opt = optim.SGD(model.parameters(), lr=1)
    opt = SketchedOptimizer(opt, workers)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion = SketchedLoss(criterion, workers)
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda_step])
    accuracy = Correct().to(device)
#lambda_step = lambda t: np.interp([t/len(train_loader)], [0, 5, args.epochs], [0, 0.4, 0])[0]/args.batch_size
#scheduler = optim.lr_scheduler.LambdaLR(sketched_opt, lr_lambda=[lambda_step])
class FakeSched(object):
    def __init__(self):
        self.x = 0
    def step(self):
        self.x += 1
    def get_lr(self):
        return [self.x]
#scheduler = FakeSched()
#train(sketched_model, sketched_opt, scheduler, sketched_criterion, accuracy,
#    train_loader, val_loader, args.epochs, loggers=(TableLogger(), TSVLogger()), timer=timer)
#model = model_cls(**model_config).to(device)
#opt = optim.SGD(model.parameters(), lr=1)
#scheduler = FakeSched()
print('Finished in {:.2f} seconds'.format(timer()))
if args.fed:
    train_fed(model, opt, scheduler, criterion, accuracy, 
        train_loader, val_loader, fed_params,
        loggers=(TableLogger(), TSVLogger()), timer=timer)
else:
    train(model, opt, scheduler, criterion, accuracy, 
        train_loader, val_loader, sketched_params,
        loggers=(TableLogger(), TSVLogger()), timer=timer)
