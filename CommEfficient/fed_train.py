import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from minimal import Net, Correct, union, PiecewiseLinear, Timer, TableLogger
from functions import FedCommEffModel, FedCommEffOptimizer, \
        FedCommEffCriterion, FedCommEffMetric
from utils import make_logdir
from gen_data import gen_data
from data_utils import get_data_loaders
from utils import parse_args

import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
global start_idx 
start_idx = 0

def train(model, opt, lr_scheduler, train_loader, val_loader,
          args, writer, loggers=(), timer=None):
    timer = timer or Timer()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    run = run_batches
    if isinstance(train_loader, np.ndarray):
        run = run_batches_fed
    for epoch in range(args.num_epochs):
        train_loss, train_acc = run(model, opt, lr_scheduler,
            train_loader, True, args)
        train_time = timer()
        val_loss, val_acc = run_batches(model, None, None,
            val_loader, False, args)
        val_time = timer()
        epoch_stats = {
            'train_time': train_time,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': val_loss,
            'test_acc': val_acc,
            'total_time': timer.total_time,
        }
        lr = lr_scheduler.get_lr()[0]
        if args.grad_reduction == "sum":
            lr = lr * batch_size
        elif args.grad_reduction in ["median", "mean"]:
            lr = lr * (batch_size / args.num_workers)
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

def run_batches(model, opt, lr_scheduler, loaders,
                training, args):
    participation = args.participation
    num_data = args.num_data
    num_clients = args.num_clients
    device = args.device
    clients = np.arange(num_clients)
    batch_size = args.batch_size
    num_workers = args.num_workers
    model.train(training)
    losses = []
    accs = []

    if training:
        global start_idx
        for batch_idx, batch in enumerate(loaders):
            inputs, targets = batch
            start_idx = start_idx % num_clients
            end_idx = start_idx + num_workers
            idx = np.random.choice(clients,
                num_workers, replace=False)
            #print(f"Selecting randomly {idx}")
            #idx = np.arange(start_idx, end_idx)
            #print(f"Selecting in order {idx}")
            minibatches = []
            for i, _ in enumerate(idx):
                start = i * batch_size // num_workers
                end = (i+1) * batch_size // num_workers
                in_batch = inputs[start:end]
                target_batch = targets[start:end]
                minibatch = [in_batch, target_batch]
                minibatches.append(minibatch)
            loss, acc = model(minibatches, idx)
            if args.use_local_sched:
                for _ in range(args.num_local_iters):
                    lr_scheduler.step()
            else:
                lr_scheduler.step()
            opt.step(idx)
            batch_loss = loss
            #print("batch_loss", batch_loss.mean())
            batch_acc = acc
            losses.append(batch_loss)
            accs.append(batch_acc)
            start_idx = end_idx
            if args.do_test:
                break

    else:
        for batch_idx, batch in enumerate(loaders):
            idx = np.arange(num_workers)
            inputs, targets = batch
            minibatches = []
            batch_len = targets.size()[0]
            indices = []
            for i, _ in enumerate(idx):
                start = i * batch_size // num_workers
                end = (i+1) * batch_size // num_workers
                if start > batch_len:
                    break
                in_batch = inputs[start:end]
                target_batch = targets[start:end]
                minibatch = [in_batch, target_batch]
                minibatches.append(minibatch)
                indices.append(i)
            #print(f"Batch sizes: {[m[1].size() for m in minibatches]}")
            if len(minibatches) > 0:
                loss, acc = model(minibatches, indices)
                batch_loss = loss
                batch_acc = acc
                losses.append(np.mean(batch_loss))
                accs.append(np.mean(batch_acc))
            """
            if args.do_test:
                break
            """
    return np.mean(losses), np.mean(accs)

#@profile
def run_batches_fed(model, opt, lr_scheduler, loaders, training, args):
    participation = args.participation
    num_data = args.num_data
    num_clients = args.num_clients
    device = args.device
    clients = np.arange(num_clients)
    batch_size = args.batch_size
    num_workers = args.num_workers
    n_iters = num_data // batch_size
    model.train(training)
    losses = []
    accs = []

    if training:
        global start_idx
        opt.step(None, True)
        for batch_idx in range(n_iters):
            idx = np.random.choice(clients,
                num_workers, replace=False)
            start_idx = start_idx % num_workers
            end_idx = start_idx + num_workers
            #print(f"Selecting randomly {idx}")
            #idx = np.arange(start_idx, end_idx)
            #print(f"Selecting in order {idx}")
            client_loaders = loaders[idx]
            minibatches = [loader.next_batch() for loader in client_loaders]
            loss, acc = model(minibatches, idx)
            if args.use_local_sched:
                for _ in range(args.num_local_iters):
                    lr_scheduler.step()
            else:
                lr_scheduler.step()
            opt.step(idx)
            model.zero_grad()
            batch_loss = loss
            #print("batch_loss", batch_loss.mean())
            batch_acc = acc
            losses.append(batch_loss)
            accs.append(batch_acc)
            start_idx = end_idx
            if args.do_test:
                break

    return np.mean(losses), np.mean(accs)

if __name__ == "__main__":
    args = parse_args(default_lr=0.4)
    timer = Timer()

    # model class and config
    model_cls = Net
    model_config = {}
    if args.do_test:
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


    # make data loaders
    train_loader, val_loader = gen_data(args)
    loader_len = args.num_data // args.batch_size

    # set up learning rate stuff
    lr_schedule = PiecewiseLinear([0, args.pivot_epoch, args.num_epochs],
                                  [0, args.lr_scale, 0])
    # grad_reduction only controlls how gradients from different
    # workers are combined
    # so the lr is multiplied by num_workers for mean and median
    if args.grad_reduction == "sum":
        lambda_step = lambda step: (lr_schedule(step / loader_len)
                                    / args.batch_size)
    else:
        lambda_step = lambda step: (lr_schedule(step / loader_len)
                                    / args.batch_size
                                    * args.num_workers)

    # instantiate ALL the things
    model = model_cls(**model_config)
    opt = optim.SGD(model.parameters(), lr=1)
    # even for median or mean, each worker still sums gradients locally
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    accuracy = Correct()

    # FedComm-ify everything
    criterion = FedCommEffCriterion(criterion, args)
    accuracy = FedCommEffMetric(accuracy, args)
    model = FedCommEffModel(model, args)
    opt = FedCommEffOptimizer(opt, args)

    lr_scheduler = optim.lr_scheduler.LambdaLR(opt,
                                               lr_lambda=[lambda_step])

    # set up output
    log_dir = make_logdir(args)
    writer = SummaryWriter(log_dir=log_dir)
    print('Finished initializing in {:.2f} seconds'.format(timer()))

    # and do the training
    train(model, opt, lr_scheduler, train_loader, val_loader, args,
          writer, loggers=(TableLogger(),), timer=timer)
