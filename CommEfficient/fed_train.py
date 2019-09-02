import os
import math
import torch
import torchvision
import numpy as np
import ray
import torch.optim as optim
import torch.nn as nn
from minimal import Net, cifar10, Correct, union, PiecewiseLinear, \
        Timer, TableLogger, normalise, pad, transpose, \
        Crop, FlipLR, Cutout, Transform, Batches, TSVLogger
from sketched_classes import SketchedModel, SketchedLoss, SketchedWorker, \
        SketchedOptimizer
from fed_sketched_classes import FedSketchedModel, FedSketchedLoss, \
        FedSketchedOptimizer, FedSketchedWorker
from fed_param_server import FedParamServer
from functions import FedCommEffModel, FedCommEffOptimizer, \
        FedCommEffCriterion, FedCommEffAccuracy
from data_utils import get_data_loaders, hp_default
DATA_PATH = 'sample_data'
GPUS_PER_WORKER = 0.8
GPUS_PER_PARAM_SERVER = 0.8

def train(model, opt, scheduler, criterion, 
    accuracy, train_loader, val_loader, 
    params, loggers=(), timer=None):
    timer = timer or Timer()
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    run = run_batches_fed
    if params["functional"]:
        run = run_batches_functional
    for epoch in range(args.epochs):
        train_loss, train_acc = run(model, opt, scheduler, 
            criterion, accuracy, train_loader, True, params)
        train_time = timer()
        val_loss, val_acc = run(model, None, scheduler,
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
        ray.timeline(filename="/tmp/timeline.json")
    return summary

def run_batches(model, opt, scheduler, criterion, 
    accuracy, loader, training, fed_params):
    model.train(training)
    losses = []
    accs = []
    w_idx = [0]
    for idx, batch in enumerate(loader):
        inputs = batch["input"]
        targets = batch["target"]
        if training:
            inputs = [inputs]
            targets = [targets]
            outs = model(inputs, w_idx)
            batch_loss = criterion(outs, targets, w_idx)
            print("batch_loss.mean()", batch_loss.mean())
            opt.zero_grad(w_idx)
            batch_loss.backward()
            opt.step(w_idx)
            scheduler.step()
            batch_acc = accuracy(
                    torch.cat(ray.get(outs), dim=0),
                    targets[0].to(args.device)
                ).float().mean().cpu().numpy()
        else:
            outs = model(inputs)
            batch_loss = criterion(outs, targets)
            batch_acc = accuracy(
                    ray.get(outs),
                    targets.to(args.device)
                ).float().mean().cpu().numpy()
        losses.append(batch_loss.mean())
        accs.append(batch_acc)
    return np.mean(losses), np.mean(accs)

def run_batches_functional(model, opt, scheduler, criterion, 
    accuracy, loaders, training, params):
    participation = params['participation']
    DATA_LEN = params['DATA_LEN']
    n_clients = params['n_clients']
    device = params['device']
    clients = np.arange(n_clients)
    batch_size = params['batch_size']
    c = (DATA_LEN/n_clients)/batch_size
    n_clients_to_select = int(n_clients * participation)
    model.train(training)
    losses = []
    accs = []

    if training:
        start_idx = 0
        for batch_idx, batch in enumerate(loaders):
            inputs = batch["input"]
            targets = batch["target"]
            start_idx = start_idx % n_clients
            end_idx = start_idx + n_clients_to_select
            #idx = np.random.choice(clients, 
            #    n_clients_to_select, replace=False)
            idx = np.arange(start_idx, end_idx)
            minibatches = []
            for i, _ in enumerate(idx):
                start = i * batch_size // n_clients_to_select
                end = (i+1) * batch_size // n_clients_to_select
                in_batch = inputs[start:end]
                target_batch = targets[start:end]
                minibatch = [in_batch, target_batch]
                minibatches.append(minibatch)
            outs, loss, acc, grads = model(minibatches, idx)
            scheduler.step()
            opt.step(grads, idx)
            batch_loss = ray.get(loss)
            batch_acc = ray.get(acc)
            losses.append(batch_loss)
            accs.append(batch_acc)
            start_idx = end_idx
            if params['test']:
                break

    else:
        for batch_idx, batch in enumerate(loaders):
            inputs, targets = batch["input"], batch["target"]
            minibatch = [inputs, targets]
            idx = [0]
            outs, loss, acc = model(minibatch, idx)
            batch_loss = ray.get(loss)
            batch_acc = ray.get(acc)
            losses.append(batch_loss)
            accs.append(batch_acc)
            if params['test']:
                break
    return np.mean(losses), np.mean(accs)

def run_batches_fed(model, opt, scheduler, criterion, 
    accuracy, loaders, training, fed_params):
    participation = fed_params['participation']
    DATA_LEN = fed_params['DATA_LEN']
    n_clients = fed_params['n_clients']
    clients = np.arange(n_clients)
    batch_size = fed_params['batch_size']
    c = (DATA_LEN/n_clients)/batch_size
    n_clients_to_select = int(n_clients * participation)
    model.train(training)
    losses = []
    accs = []

    if training:
        for idx, batch in enumerate(loaders):
            inputs = batch["input"]
            targets = batch["target"]
            idx = np.random.choice(clients, 
                n_clients_to_select, replace=False)
            input_minibatches = []
            target_minibatches = []
            for i, _ in enumerate(idx):
                start = i * batch_size // n_clients_to_select
                end = (i+1) * batch_size // n_clients_to_select
                input_minibatches.append(inputs[start:end])
                target_minibatches.append(targets[start:end])
            outs = model(input_minibatches, idx)
            batch_loss = criterion(outs, target_minibatches, idx)
            opt.zero_grad(idx)
            batch_loss.backward()
            opt.step(idx)
            scheduler.step()
            losses.append(batch_loss.mean())
            # TODO: Fix train acc calculation
            o = torch.cat(ray.get(outs), dim=0)
            batch_acc = accuracy(o, 
                                 torch.cat(target_minibatches).to(args.device)
                                ).float().mean().cpu().numpy()
            accs.append(batch_acc)

    else:
        for idx, batch in enumerate(loaders):
            inputs, targets = batch["input"], batch["target"]
            outs = model(inputs)
            batch_loss = criterion(outs, targets)
            losses.append(batch_loss.mean())
            batch_acc = accuracy(ray.get(outs), 
                    targets.to(args.device)).float().mean().cpu().numpy()
            accs.append(batch_acc)
    return np.mean(losses), np.mean(accs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=50000)
    parser.add_argument("-p2", type=int, default=4)
    parser.add_argument("-cols", type=int, default=500000)
    parser.add_argument("-rows", type=int, default=5)
    parser.add_argument("-num_blocks", type=int, default=20)
    parser.add_argument("-batch_size", type=int, default=512)
    parser.add_argument("-nesterov", action="store_true")
    parser.add_argument("-momentum", type=float, default=0.9)
    parser.add_argument("-epochs", type=int, default=24)
    parser.add_argument("-test", action="store_true")
    parser.add_argument("-fed", action="store_true")
    parser.add_argument("-sketch", action="store_true")
    parser.add_argument("-sketch_down", action="store_true")
    parser.add_argument("-functional", action="store_true")
    parser.add_argument("-virtual_momentum", action="store_true")
    parser.add_argument("-momentum_sketch", action="store_true")
    parser.add_argument("-topk_down", action="store_true")
    parser.add_argument("-clients", type=int, default=1)
    parser.add_argument("-participation", type=float, default=1.0)
    parser.add_argument("-device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

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
        args.batch_size = args.clients
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
        "n_clients_per_round": int(args.clients * args.participation),
        # optimizer params
        "lr": 1, #assumes scheduler accounts for this lr
        "momentum": args.momentum,
        "weight_decay": 5e-4*args.batch_size,
        "nesterov": args.nesterov,
        "dampening": 0,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "DATA_LEN": 50000, # specific to CIFAR10
        # algorithmic params
        "sketch": args.sketch,
        "sketch_down": args.sketch_down,
        "topk_down": args.topk_down,
        "functional": args.functional,
        "virtual_momentum": args.virtual_momentum,
        "momentum_sketch": args.momentum_sketch,
    }
    """
    if args.fed:
        train_loader, central_train_loader, val_loader, stats = get_data_loaders(params, verbose=True)

    else:
    """
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
    ray.init(
            redis_password="sketched_sgd", 
            object_store_memory=int(6e10),
            )
    lr_schedule = PiecewiseLinear([0, 5, args.epochs], [0, 0.4, 0])
    lambda_step = lambda step: lr_schedule(step/len(train_loader))/args.batch_size
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if args.functional:
        model = FedCommEffModel(model_cls, model_config, params)
        opt = torch.optim.SGD(model.parameters(), lr=1)
        opt = FedCommEffOptimizer(opt, params)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        criterion = FedCommEffCriterion(criterion, params)
        accuracy = Correct()
        accuracy = FedCommEffAccuracy(accuracy, params)
    else:
        workers = [FedSketchedWorker.remote(params
            ) for _ in range(args.clients)]
        param_server = FedParamServer.remote(params)
        model = FedSketchedModel(model_cls, model_config, workers, 
                param_server, params)
        opt = optim.SGD(model.parameters(), lr=1)
        opt = FedSketchedOptimizer(opt, workers, param_server, model)
        criterion = FedSketchedLoss(criterion, workers, param_server, model)
        accuracy = Correct().to(args.device)
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda_step])

    print('Finished initializing in {:.2f} seconds'.format(timer()))
    train(model, opt, scheduler, criterion, accuracy, 
        train_loader, val_loader, params,
        loggers=(TableLogger(), TSVLogger()), timer=timer)
