import os
import math
import torch
import torchvision
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from minimal import Net, cifar10, Correct, union, PiecewiseLinear, \
        Timer, TableLogger, normalise, pad, transpose, \
        Crop, FlipLR, Cutout, Transform, Batches, TSVLogger
from gen_data import gen_data
"""
from sketched_classes import SketchedModel, SketchedLoss, SketchedWorker, \
        SketchedOptimizer
from fed_sketched_classes import FedSketchedModel, FedSketchedLoss, \
        FedSketchedOptimizer, FedSketchedWorker
from fed_param_server import FedParamServer
"""
from functions import FedCommEffModel, FedCommEffOptimizer, \
        FedCommEffCriterion, FedCommEffAccuracy, make_logdir
from data_utils import get_data_loaders, hp_default
import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
DATA_PATH = 'sample_data'
GPUS_PER_WORKER = 0.8
GPUS_PER_PARAM_SERVER = 0.8
global start_idx 
start_idx = 0

def train(model, opt, scheduler, train_loader, val_loader, 
    params, writer, loggers=(), timer=None):
    timer = timer or Timer()
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    run = run_batches
    if isinstance(train_loader, np.ndarray):
        run = run_batches_fed
    for epoch in range(args.epochs):
        train_loss, train_acc = run(model, opt, scheduler, 
            train_loader, True, params)
        train_time = timer()
        val_loss, val_acc = run_batches(model, None, None,
            val_loader, False, params)
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
        lr = scheduler.get_lr()[0]
        if params["grad_reduce"] == "sum":
            lr = lr * batch_size
        elif params["grad_reduce"] in ["median", "mean"]:
            lr = lr * (batch_size / params['n_workers'])
        summary = union({'epoch': epoch+1, 
            'lr': lr},
            epoch_stats)
        for logger in loggers:
            logger.append(summary)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('Time/train', train_time, epoch)
        writer.add_scalar('Time/val', val_time, epoch)
        writer.add_scalar('Time/total', timer.total_time, epoch)
        writer.add_scalar('Lr', lr, epoch)
    return summary

def run_batches(model, opt, scheduler, loaders, 
        training, params): 
    participation = params['participation']
    DATA_LEN = params['DATA_LEN']
    n_clients = params['n_clients']
    device = params['device']
    clients = np.arange(n_clients)
    batch_size = params['batch_size']
    n_workers = params['n_workers']
    model.train(training)
    losses = []
    accs = []

    if training:
        global start_idx
        for batch_idx, batch in enumerate(loaders):
            inputs, targets = batch
            start_idx = start_idx % n_clients
            end_idx = start_idx + n_workers
            idx = np.random.choice(clients, 
                n_workers, replace=False)
            #print(f"Selecting randomly {idx}")
            #idx = np.arange(start_idx, end_idx)
            #print(f"Selecting in order {idx}")
            minibatches = []
            for i, _ in enumerate(idx):
                start = i * batch_size // n_workers
                end = (i+1) * batch_size // n_workers
                in_batch = inputs[start:end]
                target_batch = targets[start:end]
                minibatch = [in_batch, target_batch]
                minibatches.append(minibatch)
            outs, loss, acc = model(minibatches, idx)
            scheduler.step()
            opt.step(idx)
            batch_loss = loss
            #print("batch_loss", batch_loss.mean())
            batch_acc = acc
            losses.append(batch_loss)
            accs.append(batch_acc)
            start_idx = end_idx
            if params['test']:
                break

    else:
        for batch_idx, batch in enumerate(loaders):
            idx = np.arange(params["n_workers"])
            inputs, targets = batch
            minibatches = []
            batch_len = targets.size()[0]
            indices = []
            for i, _ in enumerate(idx):
                start = i * batch_size // n_workers
                end = (i+1) * batch_size // n_workers
                if end > batch_len:
                    break
                in_batch = inputs[start:end]
                target_batch = targets[start:end]
                minibatch = [in_batch, target_batch]
                minibatches.append(minibatch)
                indices.append(i)
            #print(f"Batch sizes: {[m[1].size() for m in minibatches]}")
            outs, loss, acc = model(minibatches, indices)
            batch_loss = loss
            batch_acc = acc
            losses.append(np.mean(batch_loss))
            accs.append(np.mean(batch_acc))
            """
            if params['test']:
                break
            """
    return np.mean(losses), np.mean(accs)

def run_batches_fed(model, opt, scheduler, loaders, 
        training, params): 
    participation = params['participation']
    DATA_LEN = params['DATA_LEN']
    n_clients = params['n_clients']
    device = params['device']
    clients = np.arange(n_clients)
    batch_size = params['batch_size']
    n_workers = params['n_workers']
    n_iters = DATA_LEN // batch_size
    model.train(training)
    losses = []
    accs = []

    if training:
        global start_idx
        for batch_idx in range(n_iters):
            idx = np.random.choice(clients, 
                n_workers, replace=False)
            start_idx = start_idx % n_workers
            end_idx = start_idx + n_workers
            #print(f"Selecting randomly {idx}")
            idx = np.arange(start_idx, end_idx)
            #print(f"Selecting in order {idx}")
            client_loaders = loaders[idx]
            minibatches = [loader.next_batch() for loader in client_loaders]
            outs, loss, acc = model(minibatches, idx)
            scheduler.step()
            opt.step(idx)
            batch_loss = loss
            #print("batch_loss", batch_loss.mean())
            batch_acc = acc
            losses.append(batch_loss)
            accs.append(batch_acc)
            start_idx = end_idx
            if params['test']:
                break

    return np.mean(losses), np.mean(accs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=50000)
    parser.add_argument("--p2", type=int, default=4)
    parser.add_argument("--cols", type=int, default=500000)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--num_blocks", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=24)
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
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--balancedness", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fed", action="store_true")
    parser.add_argument("--DATA_LEN", type=int, default=50000)
    args = parser.parse_args()
    args.workers = int(args.clients * args.participation)

    timer = Timer()
    model_cls = Net
    model_config = {}
    if args.test:
        model_config = {
            'channels': {'prep': 1, 'layer1': 1,
            'layer2': 1, 'layer3': 1},
        }
        args.num_cols = 10
        args.num_rows = 1
        args.k = 10
        args.p2 = 1 
        #args.batch_size = args.clients
    else:
        model_config = {
                'channels': {'prep': 64, 'layer1': 128,
                'layer2': 256, 'layer3': 512},
        }

    params = {
        "device": args.device,
        "test": args.test,
        # sketching params
        "k": args.k,
        "p2": args.p2,
        "num_cols": args.cols,
        "num_rows": args.rows,
        "num_blocks": args.num_blocks,
        # federation params
        "n_clients": args.clients,
        "participation": args.participation,
        "n_workers": args.workers,
        # optimizer params
        "lr": 1, #assumes scheduler accounts for this lr
        "momentum": args.momentum,
        "weight_decay": 5e-4*args.batch_size,
        "nesterov": args.nesterov,
        "dampening": 0,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_reduce": args.grad_reduce,
        "DATA_LEN": args.DATA_LEN,
        # algorithmic params
        "mode": args.mode,
        "topk_down": args.topk_down,
        "momentum_type": args.momentum_type,
        "error_type": args.error_type,
    }

    train_loader, val_loader = gen_data(args)
    loader_len = args.DATA_LEN // args.batch_size
    print('Initializing everything')
    lr_schedule = PiecewiseLinear([0, 5, args.epochs], [0, 0.4, 0])
    if args.grad_reduce == "sum":
        lambda_step = lambda step: lr_schedule(step/loader_len)/args.batch_size
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    else:
        lambda_step = lambda step: lr_schedule(step/loader_len)/(args.batch_size/params['n_workers'])
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    criterion = FedCommEffCriterion(criterion, params)
    accuracy = Correct()
    accuracy = FedCommEffAccuracy(accuracy, params)
    model = FedCommEffModel(model_cls, model_config, params)
    opt = torch.optim.SGD(model.parameters(), lr=1)
    opt = FedCommEffOptimizer(opt, params)
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda_step])

    log_dir = make_logdir(params)
    writer = SummaryWriter(log_dir=log_dir)
    print('Finished initializing in {:.2f} seconds'.format(timer()))
    tsv = TSVLogger()
    train(model, opt, scheduler, 
        train_loader, val_loader, params, writer,
        loggers=(TableLogger(), tsv), timer=timer)
    print(str(tsv))
