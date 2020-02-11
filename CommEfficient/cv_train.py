import torch
import numpy as np
import math
import os
import time
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

import models
from fed_aggregator import FedModel, FedOptimizer
from utils import make_logdir, union, Timer, TableLogger, parse_args
from utils import PiecewiseLinear, Exp, num_classes_of_dataset, steps_per_epoch
from data_utils import FedSampler, FedCIFAR10, FedImageNet, FedCIFAR100, FedEMNIST
from data_utils import cifar10_train_transforms, cifar10_test_transforms
from data_utils import cifar100_train_transforms, cifar100_test_transforms
from data_utils import imagenet_train_transforms, imagenet_val_transforms
from data_utils import femnist_train_transforms, femnist_test_transforms

import torch.multiprocessing as multiprocessing

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
    """
    if args.do_mixup:
        return compute_loss_mixup(model, batch, args)
    else:
    """
    return compute_loss_ce(model, batch, args)

def compute_loss_val(model, batch, args):
    return compute_loss_ce(model, batch, args)

def train(model, opt, lr_scheduler, train_loader, test_loader,
          args, writer, loggers=(), timer=None):
    timer = timer or Timer()

    total_download = 0
    total_upload = 0
    if args.eval_before_start:
        # val
        test_loss, test_acc, _, _ = run_batches(
                model, None, None, test_loader, False, args
            )
        test_time = timer()
        print("Test acc at epoch 0: {:0.4f}".format(test_acc))
    # ceil in case num_epochs in case we want to do a
    # fractional number of epochs
    for epoch in range(math.ceil(args.num_epochs)):
        if epoch == math.ceil(args.num_epochs) - 1:
            epoch_fraction = args.num_epochs - epoch
        else:
            epoch_fraction = 1
        # train
        train_loss, train_acc, download, upload = run_batches(
                model, opt, lr_scheduler, train_loader,
                True, epoch_fraction, args
            )
        if train_loss is np.nan:
            print("TERMINATING TRAINING DUE TO NAN LOSS")
            return

        train_time = timer()
        download_mb = download.sum().item() / (1024*1024)
        upload_mb = upload.sum().item() / (1024*1024)
        total_download += download_mb
        total_upload += upload_mb

        # val
        test_loss, test_acc, _, _ = run_batches(
                model, None, None, test_loader, False, 1, args
            )
        test_time = timer()
        # report epoch results
        try:
            rounded_down = round(download_mb)
        except:
            rounded_down = np.nan
        try:
            rounded_up = round(upload_mb)
        except:
            rounded_up = np.nan
        epoch_stats = {
            'train_time': train_time,
            'train_loss': train_loss,
            'train_acc':  train_acc,
            'test_loss':  test_loss,
            'test_acc':   test_acc,
            'down (MiB)': rounded_down,
            'up (MiB)': rounded_up,
            'total_time': timer.total_time,
        }
        lr = lr_scheduler.get_last_lr()[0]
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

    print("Total Download (MiB): {:0.2f}".format(total_download))
    print("Total Upload (MiB): {:0.2f}".format(total_upload))
    print("Avg Download Per Client: {:0.2f}".format(
        total_download / train_loader.dataset.num_clients
    ))
    print("Avg Upload Per Client: {:0.2f}".format(
        total_upload / train_loader.dataset.num_clients
    ))
    return summary

#@profile
def run_batches(model, opt, lr_scheduler, loader,
                training, epoch_fraction, args):
    if not training and epoch_fraction != 1:
        raise ValueError("Must do full epochs for val")
    if epoch_fraction > 1 or epoch_fraction <= 0:
        msg = "Invalid epoch_fraction {}.".format(epoch_fraction)
        msg += " Should satisfy 0 < epoch_fraction <= 1"
        raise ValueError(msg)

    model.train(training)
    losses = []
    accs = []

    client_download = None
    client_upload = None
    start_time = 0
    if training:
        num_clients = loader.dataset.num_clients
        client_download = torch.zeros(num_clients)
        client_upload = torch.zeros(num_clients)
        spe = steps_per_epoch(args.local_batch_size, loader.dataset,
                              args.num_workers)
        for i, batch in enumerate(loader):
            # only carry out an epoch_fraction portion of the epoch
            if i > spe * epoch_fraction:
                break

            lr_scheduler.step()

            if lr_scheduler.get_last_lr()[0] == 0:
                # hack to get the starting LR right for fedavg
                print("HACK STEP")
                opt.step()

            if args.local_batch_size == -1:
                expected_num_clients = args.num_workers
                if torch.unique(batch[0]).numel() < expected_num_clients:
                    # skip if there weren't enough clients left
                    msg = "SKIPPING BATCH: NOT ENOUGH CLIENTS ({} < {})"
                    print(msg.format(torch.unique(batch[0]).numel(),
                                     expected_num_clients))
                    continue
            else:
                expected_numel = args.num_workers * args.local_batch_size
                if batch[0].numel() < expected_numel:
                    # skip incomplete batches
                    msg = "SKIPPING BATCH: NOT ENOUGH DATA ({} < {})"
                    print(msg.format(batch[0].numel(), expected_numel))
                    continue

            loss, acc, download, upload = model(batch)
            if np.any(np.isnan(loss)):
                print(f"LOSS OF {np.mean(loss)} IS NAN, TERMINATING TRAINING")
                return np.nan, np.nan, np.nan, np.nan

            client_download += download
            client_upload += upload

            opt.step()
            #model.zero_grad()
            losses.extend(loss)
            accs.extend(acc)
            if args.dataset_name == "EMNIST":
                lr = lr_scheduler.get_last_lr()[0]
                print("LR: {:0.5f}, Loss: {:0.5f}, Acc: {:0.5f}, Time: {:0.2f}".format(
                        lr, loss.mean().item(), acc.mean().item(), time.time() - start_time
                     ))
                start_time = time.time()
            if args.do_test:
                break
    else:
        for batch in loader:
            if batch[0].numel() < args.valid_batch_size:
                print("SKIPPING VAL BATCH: TOO SMALL")
                continue
            loss, acc = model(batch)
            losses.extend(loss)
            accs.extend(acc)
            if args.do_test:
                break

    return np.mean(losses), np.mean(accs), client_download, client_upload

def get_data_loaders(args):
    train_transforms, val_transforms = {
     "ImageNet": (imagenet_train_transforms, imagenet_val_transforms),
     "CIFAR10": (cifar10_train_transforms, cifar10_test_transforms),
     "CIFAR100": (cifar100_train_transforms, cifar100_test_transforms),
     "EMNIST": (femnist_train_transforms, femnist_test_transforms),
    }[args.dataset_name]

    dataset_class = globals()["Fed" + args.dataset_name]
    train_dataset = dataset_class(args.dataset_dir, args.dataset_name, train_transforms,
                                  args.do_iid, args.num_clients,
                                  train=True, download=True)
    test_dataset = dataset_class(args.dataset_dir, args.dataset_name, val_transforms,
                                 train=False, download=False)

    train_sampler = FedSampler(train_dataset,
                               args.num_workers,
                               args.local_batch_size)

    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_sampler,
                              num_workers=args.train_dataloader_workers)
                              #multiprocessing_context="spawn",
                              #pin_memory=True)
    test_batch_size = args.valid_batch_size * args.num_workers
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=args.val_dataloader_workers)
                             #multiprocessing_context="spawn",
                             #pin_memory=True)
    print(len(train_loader), len(test_loader))

    return train_loader, test_loader

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    print("MY PID:", os.getpid())
    """
    import cProfile
    import sys
    # if check avoids hackery when not profiling
    # Optional; hackery *seems* to work fine even when not profiling,
    # it's just wasteful
    if sys.modules['__main__'].__file__ == cProfile.__file__:
        # Imports you again (does *not* use cache or execute as __main__)
        import cv_train
        # Replaces current contents with newly imported stuff
        globals().update(vars(cv_train))
        # Ensures pickle lookups on __main__ find matching version
        sys.modules['__main__'] = cv_train
    """

    # fixup
    #args = parse_args(default_lr=0.4)

    # fixup_resnet50
    #args = parse_args(default_lr=0.002)

    # fixupresnet9
    #args = parse_args(default_lr=0.06)

    args = parse_args()

    print(args)

    timer = Timer()

    # reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # model class and config
    if args.do_test:
        model_config = {
            'channels': {'prep': 1, 'layer1': 1,
                         'layer2': 1, 'layer3': 1},
        }
        args.num_cols = 10
        args.num_rows = 1
        args.k = 10
    else:
        model_config = {
                'channels': {'prep': 64, 'layer1': 128,
                             'layer2': 256, 'layer3': 512},
        }
    if args.do_finetune:
        num_classes = num_classes_of_dataset(args.finetuned_from)
        num_new_classes = num_classes_of_dataset(args.dataset_name)
    else:
        num_classes = num_classes_of_dataset(args.dataset_name)
        num_new_classes = None

    model_config.update({"num_classes": num_classes,
                         "new_num_classes": num_new_classes})
    model_config.update({"bn_bias_freeze": args.do_finetune,
                         "bn_weight_freeze": args.do_finetune})
    if args.dataset_name == "EMNIST":
        model_config["initial_channels"] = 1

    # comment out for Fixup
    model_config["do_batchnorm"] = args.do_batchnorm

    # make data loaders
    train_loader, test_loader = get_data_loaders(args)

    # instantiate ALL the things
    model_cls = getattr(models, args.model)
    model = model_cls(**model_config)

    if args.model[:5] == "Fixup":
        print("using fixup learning rates")
        params_bias = [p[1] for p in model.named_parameters()
                            if 'bias' in p[0]]
        params_scale = [p[1] for p in model.named_parameters()
                             if 'scale' in p[0]]
        params_other = [p[1] for p in model.named_parameters()
                             if not ('bias' in p[0] or 'scale' in p[0])]
        param_groups = [{"params": params_bias, "lr": 0.1},
                        {"params": params_scale, "lr": 0.1},
                        {"params": params_other, "lr": 1}]
    elif args.do_finetune:
        model.load_state_dict(torch.load(args.finetune_path + args.model + '.pt'))
        for param in model.parameters():
            param.requires_grad = False
        param_groups = model.finetune_parameters()
        """
        param_groups = model.parameters()
        """
    else:
        param_groups = model.parameters()
    opt = optim.SGD(param_groups, lr=1)

    model = FedModel(model, compute_loss_train, args, compute_loss_val)
    opt = FedOptimizer(opt, args)

    # set up learning rate scheduler
    # original cifar10_fast repo uses [0, 5, 24] and [0, 0.4, 0]
    lr_schedule = PiecewiseLinear([0, args.pivot_epoch, args.num_epochs],
                                  [0, args.lr_scale,                  0])

    # grad_reduction only controls how gradients from different
    # workers are combined
    # so the lr is multiplied by num_workers for both mean and median
    spe = steps_per_epoch(args.local_batch_size,
                          train_loader.dataset,
                          args.num_workers)
    lambda_step = lambda step: lr_schedule(step / spe)
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
          writer, loggers=(TableLogger(),), timer=timer)
    model.finalize()
    if args.do_checkpoint:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
        torch.save(model.state_dict(), args.checkpoint_path + args.model + '.pt')
