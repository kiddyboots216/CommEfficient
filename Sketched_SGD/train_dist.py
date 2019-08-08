import torch
import os
import collections
import logging
import glob
import re

import torch, torchvision
import numpy as np

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset

import itertools as it
import copy

from minimal import *
from sketched_optimizer import SketchedModel, SketchedWorker, SketchedLoss, SketchedOptimizer

DATA_PATH = 'sample_data'

# def get_cifar10():
#     '''Return CIFAR10 train/test data and labels as numpy arrays'''
#     data_train = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=True, download=True) 
#     data_test = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=False, download=True) 

#     x_train, y_train = data_train.train_data.transpose((0,3,1,2)), np.array(data_train.train_labels)
#     x_test, y_test = data_test.test_data.transpose((0,3,1,2)), np.array(data_test.test_labels)

#     return x_train, y_train, x_test, y_test

# def split_image_data(data, labels, n_clients=10, classes_per_client=10, shuffle=True, verbose=True, balancedness=None):
#         '''
#         Splits (data, labels) evenly among 'n_clients s.t. every client holds 'classes_per_client
#         different labels
#         data : [n_data x shape]
#         labels : [n_data (x 1)] from 0 to n_labels
#         '''
#         # constants
#         n_data = data.shape[0]
#         n_labels = np.max(labels) + 1
        
#         if balancedness >= 1.0:
#                 data_per_client = [n_data // n_clients]*n_clients
#                 data_per_client_per_class = [data_per_client[0] // classes_per_client]*n_clients
#         else:
#                 fracs = balancedness**np.linspace(0,n_clients-1, n_clients)
#                 fracs /= np.sum(fracs)
#                 fracs = 0.1/n_clients + (1-0.1)*fracs
#                 data_per_client = [np.floor(frac*n_data).astype('int') for frac in fracs]

#                 data_per_client = data_per_client[::-1]

#                 data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]

#         if sum(data_per_client) > n_data:
#                 print("Impossible Split")
#                 exit()
        
#         # sort for labels
#         data_idcs = [[] for i in range(n_labels)]
#         for j, label in enumerate(labels):
#                 data_idcs[label] += [j]
#         if shuffle:
#                 for idcs in data_idcs:
#                         np.random.shuffle(idcs)
                
#         # split data among clients
#         clients_split = []
#         c = 0
#         for i in range(n_clients):
#                 client_idcs = []
#                 budget = data_per_client[i]
#                 c = np.random.randint(n_labels)
#                 while budget > 0:
#                         take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
                        
#                         client_idcs += data_idcs[c][:take]
#                         data_idcs[c] = data_idcs[c][take:]
                        
#                         budget -= take
#                         c = (c + 1) % n_labels
                        
#                 clients_split += [(data[client_idcs], labels[client_idcs])]
        
#         def print_split(clients_split): 
#                 print("Data split:")
#                 for i, client in enumerate(clients_split):
#                         split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
#                         print(" - Client {}: {}".format(i,split))
#                 print()
                        
#         if verbose:
#                 print_split(clients_split)
                                
#         return clients_split

# class CustomImageDataset(Dataset):
#         '''
#         A custom Dataset class for images
#         inputs : numpy array [n_data x shape]
#         labels : numpy array [n_data (x 1)]
#         '''
#         def __init__(self, inputs, labels, transforms=None):
#                         assert inputs.shape[0] == labels.shape[0]
#                         self.inputs = torch.Tensor(inputs)
#                         self.labels = torch.Tensor(labels).long()
#                         self.transforms = transforms 

#         def __getitem__(self, index):
#                         img, label = self.inputs[index], self.labels[index]

#                         if self.transforms is not None:
#                                 img = self.transforms(img)

#                         return (img, label)

#         def __len__(self):
#                         return self.inputs.shape[0]
                                        

# def get_default_data_transforms(name, train=True, verbose=True):
#         transforms_train = {
#         'mnist' : transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((32, 32)),
#                 #transforms.RandomCrop(32, padding=4),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.06078,),(0.1957,))
#                 ]),
#         'fashionmnist' : transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((32, 32)),
#                 #transforms.RandomCrop(32, padding=4),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,))
#                 ]),
#         'cifar10' : transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.RandomCrop(32, padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
#         'kws' : None
#         }
#         transforms_eval = {
#         'mnist' : transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((32, 32)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.06078,),(0.1957,))
#                 ]),
#         'fashionmnist' : transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((32, 32)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,))
#                 ]),
#         'cifar10' : transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#
#         'kws' : None
#         }

#         if verbose:
#                 print("\nData preprocessing: ")
#                 for transformation in transforms_train[name].transforms:
#                         print(' -', transformation)
#                 print()

#         return (transforms_train[name], transforms_eval[name])


# def get_data_loaders(hp, verbose=True):
        
#         x_train, y_train, x_test, y_test = globals()['get_'+hp['dataset']]()

#         if verbose:
#                 print_image_data_stats(x_train, y_train, x_test, y_test)

#         transforms_train, transforms_eval = get_default_data_transforms(hp['dataset'], verbose=False)

#         split = split_image_data(x_train, y_train, n_clients=hp['n_clients'], 
#                                         classes_per_client=hp['classes_per_client'], balancedness=hp['balancedness'], verbose=verbose)

#         client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), 
#                           batch_size=hp['batch_size'], shuffle=True) for x, y in split]
#         train_loader = torch.utils.data.DataLoader(CustomImageDataset(x_train, y_train, transforms_eval), batch_size=100, shuffle=False)
#         test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False) 

#         stats = {"split" : [x.shape[0] for x, y in split]}

#         return client_loaders, train_loader, test_loader, stats

# hp_default = {
#     "dataset" : "cifar10", 
#     "net" : "logistic",

#     "iterations" : 2000,

#     "n_clients" : 4,
#     "participation_rate" : 1.0,
#     "classes_per_client" : 10,
#     "batch_size" : 1,
#     "balancedness" : 1.0,   

#     "momentum" : 0.9,


#     "compression" : ["none", {}],

#     "log_frequency" : 30,
#     "log_path" : "results/trash/"
# }

def train(model, opt, scheduler, criterion, accuracy, 
    train_loader, val_loader, epochs, loggers=(), timer=None):
    timer = timer or Timer()
    for epoch in range(args.epochs):
        train_losses, train_accs = run_batches(sketched_model, sketched_opt, 
            sketched_criterion, accuracy, train_loader, True)
        train_time = timer()
        val_losses, val_accs = run_batches(sketched_model, None, 
            sketched_criterion, accuracy, val_loader, False)
        val_time = timer()
        epoch_stats = {
            'train_time': train_time,
            'train_loss': np.mean(train_losses)
            'train_acc': np.mean(train_accs)
            'val_time': val_time,
            'val_loss': np.mean(val_loss)
            'val_acc': np.mean(val_accs)
            'total_time': timer.total_time
        }
        summary = union({'epoch': epoch+1, 'lr': scheduler.get_lr()}, epoch_stats)
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
        batch_acc = accuracy(outs, targets).mean().cpu().numpy()
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
args = parser.parse_args()

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
    "numCols": args.cols,
    "numRows": args.rows,
    "numBlocks": args.num_blocks,
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
ray.init(num_gpus=8)
model_cls = Net
model_config = {}
workers = [SketchedWorker.remote(*sketched_params) for _ in range(args.num_workers)]
sketched_model = SketchedModel(model_cls, model_config, workers)
opt = optim.SGD(sketched_model.parameters(), lr=1)
sketched_opt = SketchedOptimizer(opt, workers)
criterion = torch.nn.MSELoss(reduction='sum')
sketched_criterion = SketchedLoss(criterion, workers)
accuracy = Correct().to(device)
lambda_step = lambda t: np.interp([t], [0, 5, args.epochs], [0, 0.4, 0])
scheduler = optim.lr_scheduler.LambdaLR(sketched_opt, lr_lambda=[lambda_step])
train(sketched_model, sketched_opt, scheduler, sketched_criterion, accuracy,
    train_loader, val_loader, args.epochs, loggers=(TableLogger(), TSVLogger()), timer=timer)
