import torch
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from models import configs
import models
from fed_aggregator import FedModel, FedOptimizer
from utils import make_logdir, union, Timer, TableLogger, parse_args
from data_utils import FedSampler, FedCIFAR10, FedImageNet
from data_utils import cifar_train_transforms, cifar_test_transforms
from data_utils import imagenet_train_transforms, imagenet_val_transforms

import torch.multiprocessing as multiprocessing

from dp_functions import DPGaussianHook

#from line_profiler import LineProfiler
#import atexit
#profile = LineProfiler()
#atexit.register(profile.print_stats)

# module for computing accuracy
class Correct(torch.nn.Module):
    def forward(self, classifier, target):
        return (classifier.max(dim = 1)[1] == target).float().mean()

def criterion_helper(outputs, target, lam):
    ce = -F.log_softmax(outputs, dim=1)
    mixed = torch.zeros_like(outputs).scatter_(
                1, target.data.view(-1, 1), lam.view(-1, 1)
            )
    return (ce * mixed).sum(dim=1).mean()

def mixup_criterion(outputs, y_a, y_b, lam):
    return (criterion_helper(outputs, y_a, lam)
            + criterion_helper(outputs, y_b, 1 - lam))

# whether args.grad_reduction is median or mean,
# each worker still means gradients locally
ce_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

accuracy_metric = Correct()

def compute_loss_mixup(model, batch, args):
    images, targets = batch
    inputs, targets_a, targets_b, lam = mixup_data(
            images, targets, args.mixup_alpha,
            use_cuda="cuda" in args.device
        )
    outputs = model(inputs)
    loss = mixup_criterion(outputs, targets_a, targets_b, lam)
    pred = torch.max(outputs, 1)[1]
    correct = (lam * pred.eq(targets_a)
               + (1 - lam) * pred.eq(targets_b)).float().sum()
    accuracy = correct / targets.size()[0]
    return loss, accuracy

def compute_loss_ce(model, batch, args):
    images, targets = batch
    pred = model(images)
    loss = ce_criterion(pred, targets)
    accuracy = accuracy_metric(pred, targets)
    return loss, accuracy

def compute_loss_train(model, batch, args):
    if args.do_mixup:
        return compute_loss_mixup(model, batch, args)
    else:
        return compute_loss_ce(model, batch, args)

def compute_loss_mal(model, batch, args):
    images, targets = batch
    pred = model(images)
    loss = ce_criterion(pred, targets)
    accuracy = accuracy_metric(pred, targets)
    accuracy *= (args.local_batch_size / args.mal_targets)
    boosted_loss = args.mal_boost * loss
    return boosted_loss, accuracy

def compute_loss_val(model, batch, args):
    return compute_loss_ce(model, batch, args)

def train(model, opt, lr_scheduler, train_loader, test_loader,
          args, writer, loggers=(), timer=None, mal_loader=None):
    timer = timer or Timer()
    for epoch in range(args.num_epochs):
        epoch_stats = {}
        train_loss, train_acc = run_batches(model, opt, lr_scheduler,
                                            train_loader, True, True, args)
        test_loss, test_acc = run_batches(model, None, None,
                                          test_loader, False, False, args)
        if args.is_malicious:
            mal_loss, mal_acc = run_batches(model, opt, lr_scheduler,
                mal_loader, False, False, args)
            epoch_stats['mal_loss'] = mal_loss
            epoch_stats['mal_acc'] = mal_acc
            if args.use_tensorboard:
                writer.add_scalar('Loss/mal',   mal_loss,         epoch)
                writer.add_scalar('Acc/mal',    mal_acc,          epoch)
        train_time = timer()
        test_time = timer()
        epoch_stats.update({
            'train_time': train_time,
            'train_loss': train_loss,
            'train_acc':  train_acc,
            'test_loss':  test_loss,
            'test_acc':   test_acc,
            'total_time': timer.total_time,
        })
        lr = lr_scheduler.get_lr()[0]
        summary = union({'epoch': epoch+1,
                         'lr': lr},
                        epoch_stats)
        for logger in loggers:
            logger.append(summary)
        if args.use_tensorboard:
            writer.add_scalar('Loss/train', train_loss,       epoch)
            writer.add_scalar('Loss/test',  test_loss,        epoch)
            writer.add_scalar('Acc/train',  train_acc,        epoch)
            writer.add_scalar('Acc/test',   test_acc,         epoch)
            writer.add_scalar('Time/train', train_time,       epoch)
            writer.add_scalar('Time/test',  test_time,        epoch)
            writer.add_scalar('Time/total', timer.total_time, epoch)
            writer.add_scalar('Lr',         lr,               epoch)
    return summary

#@profile
def run_batches(model, opt, lr_scheduler, loader, training, step_scheduler, args):
    model.train(training)
    losses = []
    accs = []

    if training:
        for i, batch in enumerate(loader):
            loss, acc = model(batch)
            if args.use_local_sched:
                for _ in range(args.num_local_iters):
                    lr_scheduler.step()
            else:
                lr_scheduler.step()
            opt.step()
            #model.zero_grad()
            losses.extend(loss)
            accs.extend(acc)
            if args.do_test:
                break
    else:
        for batch in loader:
            loss, acc = model(batch)
            losses.extend(loss)
            accs.extend(acc)

    return np.mean(losses), np.mean(accs)

def get_data_loaders(args):
    train_transforms, val_transforms = {
     "ImageNet": (imagenet_train_transforms, imagenet_val_transforms),
     "CIFAR10": (cifar_train_transforms, cifar_test_transforms)
    }[args.dataset_name]

    dataset_class = globals()["Fed" + args.dataset_name]
    train_dataset = dataset_class(args, args.dataset_dir, train_transforms,
                                  args.do_iid, args.num_clients,
                                  train=True, download=True)
    test_dataset = dataset_class(args, args.dataset_dir, val_transforms,
                                 train=False, download=False)

    train_sampler = FedSampler(train_dataset,
                               args.num_workers,
                               args.local_batch_size)

    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=8,
                              pin_memory=True)
    test_batch_size = args.local_batch_size * args.num_workers
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    mal_loader = None
    if args.is_malicious:
        mal_dataset = dataset_class(args, args.dataset_dir, train_transforms,
                                  args.do_iid, args.num_clients,
                                  train=True, download=False, malicious=True)
        mal_loader = DataLoader(mal_dataset, 
                                batch_size=args.mal_targets,
                                num_workers=4, 
                                pin_memory=True)

    return train_loader, test_loader, mal_loader


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    # fixup
    #args = parse_args(default_lr=0.4)

    # fixup_resnet50
    #args = parse_args(default_lr=0.002)

    # fixupresnet9
    #args = parse_args(default_lr=0.06)

    args = parse_args()
    config_class = getattr(configs, args.model + "Config")
    config = config_class()
    config.set_args(args)
    print(args)


    timer = Timer()
    args.lr_epoch = 0.0

    #Setting numpy random seed
    np.random.seed(21)
    # model class and config
    torch.random.manual_seed(21)
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

    # comment out for Fixup
    #model_config["iid"] = args.do_iid


    # make data loaders
    # train_loader, val_loader = gen_data(args)
    # if args.is_malicious:
    #     #To-do: remove hard-coding of mal_client_idx
    #     mal_weird_loader, mal_loader = MalLoader(args)
    #     mal_client_idx = 0
    #     train_loader[mal_client_idx] = mal_weird_loader
    # loader_len = args.num_data // args.batch_size

    train_loader, test_loader, mal_loader = get_data_loaders(args)

    # instantiate ALL the things
    #model = ResNet9(**model_config)
    #opt = optim.SGD(model.parameters(), lr=1)

    model_cls = getattr(models, args.model)
    model = model_cls(**config.model_config)

    params_bias = [p[1] for p in model.named_parameters()
                        if 'bias' in p[0]]
    params_scale = [p[1] for p in model.named_parameters()
                         if 'scale' in p[0]]
    params_other = [p[1] for p in model.named_parameters()
                         if not ('bias' in p[0] or 'scale' in p[0])]
    opt = optim.SGD([
            {"params": params_bias, "lr": 0.1},
            {"params": params_scale, "lr": 0.1},
            {"params": params_other, "lr": 1}
        ], lr=1)


    hook = None
    if args.do_dp:
        hook_cls = DPGaussianHook(args)
        hook = hook_cls.client_hook

    # Fed-ify everything
    model = FedModel(model, compute_loss_train, args,
                    compute_loss_val=compute_loss_val,
                    compute_loss_mal=compute_loss_mal,
                    hook=hook)
    opt = FedOptimizer(opt, args)

    # set up learning rate stuff
    #lr_schedule = PiecewiseLinear([0, args.pivot_epoch, args.num_epochs],
    #                              [0, args.lr_scale, 0])
    lr_schedule = config.lr_schedule

    # grad_reduction only controls how gradients from different
    # workers are combined
    # so the lr is multiplied by num_workers for both mean and median
    batch_size = args.local_batch_size * args.num_workers
    steps_per_epoch = np.ceil(len(train_loader) / batch_size)
    lambda_step = lambda step: lr_schedule(step / steps_per_epoch)
    lr_scheduler = LambdaLR(opt, lr_lambda=lambda_step)

    # set up output
    log_dir = make_logdir(args)
    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None
    print('Finished initializing in {:.2f} seconds'.format(timer()))

    # and do the training
    train(model, opt, lr_scheduler, train_loader, test_loader, args,
          writer, loggers=(TableLogger(),), timer=timer, mal_loader=mal_loader)
