import warnings
warnings.filterwarnings('ignore')
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
from utils import get_param_vec, set_param_vec
from data_utils import FedSampler, FedCIFAR10, FedImageNet, FedCIFAR100, FedFEMNIST
from data_utils import cifar10_train_transforms, cifar10_test_transforms
from data_utils import cifar100_train_transforms, cifar100_test_transforms
from data_utils import imagenet_train_transforms, imagenet_val_transforms
from data_utils import femnist_train_transforms, femnist_test_transforms

import torch.multiprocessing as multiprocessing

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

# whether args.grad_reduction is median or mean,
# each worker still means gradients locally
ce_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

accuracy_metric = Correct()

def compute_loss_ce(model, batch):
    images, targets = batch
    pred = model(images)
    loss = ce_criterion(pred, targets)
    accuracy = accuracy_metric(pred, targets)
    return loss, accuracy

def compute_loss_val(model, batch):
    return compute_loss_ce(model, batch)

def get_data_loaders(args):
    val_transforms = {
     "CIFAR10": cifar10_test_transforms,
     "CIFAR100": cifar100_test_transforms,
     "FEMNIST": femnist_test_transforms,
    }[args.dataset_name]
    transforms = cifar10_test_transforms
    testset = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=False,
                                           download=True, transform=transforms)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    """
    dataset_class = globals()["Fed" + args.dataset_name]
    test_dataset = dataset_class(args, args.dataset_dir, args.dataset_name, transform=val_transforms,
                                 train=False, download=False)

    test_batch_size = args.valid_batch_size * args.num_workers
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=args.val_dataloader_workers)
                             #multiprocessing_context="spawn",
                             #pin_memory=True)
    """

    return test_loader

def model_eval(model, test_loader, args, loggers=(), timer=None):
    timer = timer or Timer()
    losses = []
    accs = []
    model.eval()
    for i, batch in enumerate(test_loader):
        loss, acc = compute_loss_val(model, batch)
        losses.append(loss.detach().item())
        accs.append(acc.item())
    summary = {'losses': np.mean(losses), 'accs': np.mean(accs)}
    for logger in loggers:
        logger.append(summary)
    return summary

if __name__ == "__main__":
    args = parse_args()

    print(args)
    # reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # instantiate ALL the things
    model_config = {
            'channels': {'prep': 64, 'layer1': 128,
                         'layer2': 256, 'layer3': 512},
    }
    model_config["do_batchnorm"] = args.do_batchnorm
    num_classes = num_classes_of_dataset(args.dataset_name)
    model_config.update({"num_classes": num_classes,})
    model_cls = getattr(models, args.model)
    model = model_cls(**model_config)
    PATH = args.checkpoint_path + args.model + str(args.mode) + str(args.robustagg) + str(0) + '.pt'
    print("Loading model from ", PATH)
    model.load_state_dict(torch.load(PATH))
    print("Model loaded...")
    for param in model.parameters():
        param.requires_grad = False
    model.dp_parameters()
    test_loader = get_data_loaders(args) 
    if args.noise_multiplier > 0:
        param_vec = get_param_vec(model)
        print("Number of grad elements ", param_vec.shape)
        noise_vec = torch.normal(mean=torch.zeros_like(param_vec), std=torch.tensor(args.noise_multiplier).float())
        #laplace = torch.distributions.laplace.Laplace(torch.zeros_like(param_vec), torch.tensor(args.noise_multiplier).float())
        #noise_vec = laplace.sample() 
        param_vec += noise_vec
        set_param_vec(model, param_vec)
        #for param in model.parameters():
        #    param.data.add_(torch.normal(torch.tensor(0).float(),torch.tensor(args.noise_multiplier).float()))
    model_eval(model, test_loader, args, loggers=(TableLogger(),))
