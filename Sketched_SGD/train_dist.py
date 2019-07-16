import torch
import os
import collections
import logging
import glob
import re

import torch, torchvision
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset

import itertools as it
import copy

DATA_PATH = 'content/sample_data'

def get_cifar10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=True, download=True) 
    data_test = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=False, download=True) 

    x_train, y_train = data_train.train_data.transpose((0,3,1,2)), np.array(data_train.train_labels)
    x_test, y_test = data_test.test_data.transpose((0,3,1,2)), np.array(data_test.test_labels)

    return x_train, y_train, x_test, y_test

def split_image_data(data, labels, n_clients=10, classes_per_client=10, shuffle=True, verbose=True, balancedness=None):
        '''
        Splits (data, labels) evenly among 'n_clients s.t. every client holds 'classes_per_client
        different labels
        data : [n_data x shape]
        labels : [n_data (x 1)] from 0 to n_labels
        '''
        # constants
        n_data = data.shape[0]
        n_labels = np.max(labels) + 1
        
        if balancedness >= 1.0:
                data_per_client = [n_data // n_clients]*n_clients
                data_per_client_per_class = [data_per_client[0] // classes_per_client]*n_clients
        else:
                fracs = balancedness**np.linspace(0,n_clients-1, n_clients)
                fracs /= np.sum(fracs)
                fracs = 0.1/n_clients + (1-0.1)*fracs
                data_per_client = [np.floor(frac*n_data).astype('int') for frac in fracs]

                data_per_client = data_per_client[::-1]

                data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]

        if sum(data_per_client) > n_data:
                print("Impossible Split")
                exit()
        
        # sort for labels
        data_idcs = [[] for i in range(n_labels)]
        for j, label in enumerate(labels):
                data_idcs[label] += [j]
        if shuffle:
                for idcs in data_idcs:
                        np.random.shuffle(idcs)
                
        # split data among clients
        clients_split = []
        c = 0
        for i in range(n_clients):
                client_idcs = []
                budget = data_per_client[i]
                c = np.random.randint(n_labels)
                while budget > 0:
                        take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
                        
                        client_idcs += data_idcs[c][:take]
                        data_idcs[c] = data_idcs[c][take:]
                        
                        budget -= take
                        c = (c + 1) % n_labels
                        
                clients_split += [(data[client_idcs], labels[client_idcs])]
        
        def print_split(clients_split): 
                print("Data split:")
                for i, client in enumerate(clients_split):
                        split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
                        print(" - Client {}: {}".format(i,split))
                print()
                        
        if verbose:
                print_split(clients_split)
                                
        return clients_split

class CustomImageDataset(Dataset):
        '''
        A custom Dataset class for images
        inputs : numpy array [n_data x shape]
        labels : numpy array [n_data (x 1)]
        '''
        def __init__(self, inputs, labels, transforms=None):
                        assert inputs.shape[0] == labels.shape[0]
                        self.inputs = torch.Tensor(inputs)
                        self.labels = torch.Tensor(labels).long()
                        self.transforms = transforms 

        def __getitem__(self, index):
                        img, label = self.inputs[index], self.labels[index]

                        if self.transforms is not None:
                                img = self.transforms(img)

                        return (img, label)

        def __len__(self):
                        return self.inputs.shape[0]
                                        

def get_default_data_transforms(name, train=True, verbose=True):
        transforms_train = {
        'mnist' : transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                #transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.06078,),(0.1957,))
                ]),
        'fashionmnist' : transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                #transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]),
        'cifar10' : transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
        'kws' : None
        }
        transforms_eval = {
        'mnist' : transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.06078,),(0.1957,))
                ]),
        'fashionmnist' : transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]),
        'cifar10' : transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#
        'kws' : None
        }

        if verbose:
                print("\nData preprocessing: ")
                for transformation in transforms_train[name].transforms:
                        print(' -', transformation)
                print()

        return (transforms_train[name], transforms_eval[name])


def get_data_loaders(hp, verbose=True):
        
        x_train, y_train, x_test, y_test = globals()['get_'+hp['dataset']]()

        if verbose:
                print_image_data_stats(x_train, y_train, x_test, y_test)

        transforms_train, transforms_eval = get_default_data_transforms(hp['dataset'], verbose=False)

        split = split_image_data(x_train, y_train, n_clients=hp['n_clients'], 
                                        classes_per_client=hp['classes_per_client'], balancedness=hp['balancedness'], verbose=verbose)

        client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), 
                          batch_size=hp['batch_size'], shuffle=True) for x, y in split]
        train_loader = torch.utils.data.DataLoader(CustomImageDataset(x_train, y_train, transforms_eval), batch_size=100, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False) 

        stats = {"split" : [x.shape[0] for x, y in split]}

        return client_loaders, train_loader, test_loader, stats

hp_default = {
    "dataset" : "cifar10", 
    "net" : "logistic",

    "iterations" : 2000,

    "n_clients" : 4,
    "participation_rate" : 1.0,
    "classes_per_client" : 10,
    "batch_size" : 1,
    "balancedness" : 1.0,   

    "momentum" : 0.9,


    "compression" : ["none", {}],

    "log_frequency" : 30,
    "log_path" : "results/trash/"
}
import torch.nn as nn
import torch.optim as optim
import copy

from minimal import *
from single_trainer import SGD_Sketched

@ray.remote(num_gpus=1.0, num_cpus=2.0)
class SketchFedServer(object):
    def __init__(self, kwargs):
        self.device = torch.device("cuda")
        self.hp = kwargs
        opt_params = {
            "lr": 0,
            "momentum": kwargs['momentum'],
            "weight_decay": 5e-4*kwargs['batch_size'],
            "nesterov": kwargs['nesterov'],
            "dampening": 0,
        }
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        model = Net().to(self.device)
        model = SketchedModel(model)
        torch.random.set_rng_state(rand_state)
        opt = optim.SGD(model.parameters(), **opt_params)
        self._sketcher_init(**{
            'opt': opt,
            'k': self.hp['k'], 
            'p2': self.hp['p2'], 
            'numCols': self.hp['numCols'], 
            'numRows': self.hp['numRows'], 
            'numBlocks': self.hp['numBlocks']})

    def _sketcher_init(self, opt,
                 k=0, p2=0, numCols=0, numRows=0, numBlocks=1):
        #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids())) 
        #print(ray.get_gpu_ids())
        # set the sketched params as instance vars
        self.p2 = p2
        self.k = k
        # initialize sketch_mask, sketch, momentum buffer and accumulated grads
        # this is D
        grad_size = 0
        sketch_mask = []
        for group in opt.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    size = torch.numel(p)
                    if p.do_sketching:
                        sketch_mask.append(torch.ones(size))
                    else:
                        sketch_mask.append(torch.zeros(size))
                    grad_size += size
        self.grad_size = grad_size
        print(f"Total dimension is {self.grad_size} using k {self.k} and p2 {self.p2}")
        self.sketch_mask = torch.cat(sketch_mask).byte().to(self.device)
        #print(f"sketch_mask.sum(): {self.sketch_mask.sum()}")
#         print(f"Make CSVec of dim{numRows}, {numCols}")
        self.sketch = CSVec(d=self.sketch_mask.sum().item(), c=numCols, r=numRows, 
                            device=self.device, nChunks=1, numBlocks=numBlocks)
        del sketch_mask


    def sim_update(self, *diff_vecs):
        diff_vecs = [diff_vec.to(self.device) for diff_vec in diff_vecs]
        self.sketch.zero()
        for diff_vec in diff_vecs:
            self.sketch += diff_vec[self.sketch_mask]
            #/len(diff_vecs)
        candidate_top_k = self.sketch.unSketch(k=self.p2*self.k)
        candidate_hh_coords = candidate_top_k.nonzero()
        hhs = [diff_vec[candidate_hh_coords] for diff_vec in diff_vecs]
        candidate_top_k[candidate_hh_coords] = torch.sum(
            torch.stack(hhs),dim=0)
        weights = self.topk(candidate_top_k, k=self.k)
        weight_update = torch.zeros(self.grad_size, device=self.device)
        weight_update[self.sketch_mask] = weights
        weight_update[~self.sketch_mask] = torch.sum(
            torch.stack(
                [diff_vec[~self.sketch_mask] for diff_vec in diff_vecs]), dim=0)
        # COMMUNICATE
        return weight_update

    def compute_hhcoords(self, *tables):
        #_, _, tables = list(zip(*lossAcc))
        self.sketch.zero()
        self.sketch.table = torch.sum(torch.stack(tables), dim=0).to(self.device)
        self.candidateTopK = self.sketch.unSketch(k=self.p2*self.k)
        self.candidateHHCoords = self.candidateTopK.nonzero()
        # COMMUNICATE
        return self.candidateHHCoords

    def average_grads(self, *grads):
        return torch.mean(torch.stack(grads), dim=0)

    def compute_update(self, *sketchesAndUnsketched):
        hhs, unsketched = list(zip(*sketchesAndUnsketched))
        self.candidateTopK[self.candidateHHCoords] = torch.sum(
            torch.stack(hhs),dim=0)
        del self.candidateHHCoords
        weights = self.topk(self.candidateTopK, k=self.k)
        del self.candidateTopK
        weightUpdate = torch.zeros(self.grad_size, device=self.device)
        weightUpdate[self.sketch_mask] = weights
        weightUpdate[~self.sketch_mask] = torch.sum(torch.stack(unsketched), dim=0)
        # COMMUNICATE
        return weightUpdate

    def topk(self, vec, k):
        """ Return the largest k elements (by magnitude) of vec"""
        ret = torch.zeros_like(vec)

        # on a gpu, sorting is faster than pytorch's topk method
        topkIndices = torch.sort(vec**2)[1][-k:]
        #_, topkIndices = torch.topk(vec**2, k)

        ret[topkIndices] = vec[topkIndices]
        return ret

@ray.remote(num_gpus=1.0, num_cpus=2.0)
class FedSketchWorker(object):
    def __init__(self, loader, worker_index, kwargs):
        self.device = torch.device("cuda")
        self.loader = loader
        self.hp = kwargs
        self.worker_index = worker_index
        print(f"Initializing worker {self.worker_index}")
        self.opt_params = {
            "lr": kwargs['lr'],
            "momentum": kwargs['momentum'],
            "weight_decay": 5e-4*kwargs['batch_size'],
            "nesterov": kwargs['nesterov'],
            "dampening": 0,
        }
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        model = Net().to(self.device)
        self.model = SketchedModel(model)
        torch.random.set_rng_state(rand_state)
        self.crit = nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.acc = Correct().to(self.device)
        step_number = 0
        self.opt = optim.SGD(self.model.parameters(), **self.param_values(step_number))
        #self.opt = 
        self._sketcher_init(**{
            'k': self.hp['k'], 
            'p2': self.hp['p2'], 
            'numCols': self.hp['numCols'], 
            'numRows': self.hp['numRows'], 
            'numBlocks': self.hp['numBlocks']})

    def param_values(self, step_number):
        #import pdb; pdb.set_trace()
        return {k: v(step_number) if callable(v) else v for k,v in self.opt_params.items()}
    
    def topk(self, vec, k):
        """ Return the largest k elements (by magnitude) of vec"""
        ret = torch.zeros_like(vec)

        # on a gpu, sorting is faster than pytorch's topk method
        topkIndices = torch.sort(vec**2)[1][-k:]
        #_, topkIndices = torch.topk(vec**2, k)

        ret[topkIndices] = vec[topkIndices]
        return ret

    def fetch_opt_params(self):
        assert len(self.opt.param_groups) == 1
        return {'lr': self.opt.param_groups[0]['lr'], 'momentum': self.opt.param_groups[0]['momentum'], 'weight_decay': self.opt.param_groups[0]['weight_decay'], 'nesterov': self.opt.param_groups[0]['nesterov'], 'dampening': self.opt.param_groups[0]['dampening']}

    def all_reduce(self, diff_vecs):
        self.apply_update(torch.mean(torch.stack(diff_vecs), dim=0))

    def all_reduce_sketched(self, *diff_vecs):
        diff_vecs = [diff_vec.to(self.device) for diff_vec in diff_vecs]
        self.sketch.zero()
        for diff_vec in diff_vecs:
            self.sketch += diff_vec[self.sketch_mask]
            #/len(diff_vecs)
        candidate_top_k = self.sketch.unSketch(k=self.p2*self.k)
        candidate_hh_coords = candidate_top_k.nonzero()
        hhs = [diff_vec[candidate_hh_coords] for diff_vec in diff_vecs]
        candidate_top_k[candidate_hh_coords] = torch.sum(
            torch.stack(hhs),dim=0)
        weights = self.topk(candidate_top_k, k=self.k)
        weight_update = torch.zeros(self.grad_size, device=self.device)
        weight_update[self.sketch_mask] = weights
        weight_update[~self.sketch_mask] = torch.sum(
            torch.stack(
                [diff_vec[~self.sketch_mask] for diff_vec in diff_vecs]), dim=0)
        self.apply_update(weight_update)

    def train_iters(self, step_number, training, iterations):
        model = self.model
        optimizer = self.opt
        if training:
            dataloader = self.loader
            opt_copy = copy.deepcopy(self.opt.param_groups)
        # else:
        #     dataloader = self.test_loader
        criterion = self.crit
        accuracy = self.acc
        running_loss = 0.0
        running_acc = 0.0
        dataloader_iterator = iter(dataloader)
        for i in range(iterations):
            try:
                inputs, targets = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                inputs, targets = next(dataloader_iterator)
            optimizer.zero_grad()
            inputs = batch["input"]
            targets = batch["target"]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = accuracy(outputs, targets)
            if training:
                step_number += 1
                optimizer.param_groups[0].update(**self.param_values(step_number))
                loss.sum().backward()
                optimizer.step()
            running_loss += loss.float().mean().detach().cpu().numpy()
            running_acc += acc.float().mean().detach().cpu().numpy()
        if training:
            return (running_loss/len(dataloader)), (running_acc/len(dataloader)), step_number, self.model_diff(opt_copy).cpu()
        else:
            return (running_loss/len(dataloader)), (running_acc/len(dataloader))

    def train_epoch(self, step_number, training):
        model = self.model
        optimizer = self.opt
        if training:
            dataloader = self.train_loader
            opt_copy = copy.deepcopy(self.opt.param_groups)
        else:
            dataloader = self.test_loader
        criterion = self.crit
        accuracy = self.acc
        running_loss = 0.0
        running_acc = 0.0
        for idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = batch["input"]
            targets = batch["target"]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = accuracy(outputs, targets)
            if training:
                step_number += 1
                optimizer.param_groups[0].update(**self.param_values(step_number))
                loss.sum().backward()
                optimizer.step()
            running_loss += loss.float().mean().detach().cpu().numpy()
            running_acc += acc.float().mean().detach().cpu().numpy()
        if training:
            return (running_loss/len(dataloader)), (running_acc/len(dataloader)), step_number, self.model_diff(opt_copy).cpu()
        else:
            return (running_loss/len(dataloader)), (running_acc/len(dataloader))

    def model_diff(self, opt_copy):
        diff_vec = []
        for group_id, param_group in enumerate(self.opt.param_groups):
            for idx, p in enumerate(param_group['params']):
                # calculate the difference between the current model and the stored model
                diff_vec.append(opt_copy[group_id]['params'][idx].data.view(-1).float() - p.data.view(-1).float())
                # reset the current model to the stored model for later
                p.data = opt_copy[group_id]['params'][idx].data
        self.diff_vec = torch.cat(diff_vec).to(self.device)
        return self.diff_vec
        #import pdb; pdb.set_trace()
        #print(f"Found a difference of {torch.sum(self.diff_vec)}")
        # return self.diff_vec
        # print(diff_vec)
        masked_diff = self.diff_vec[self.sketch_mask]
        # sketch the gradient
        self.sketch.zero()
        self.sketch += masked_diff
        del masked_diff
        # communicate only the table
        return self.sketch.table

    # def send_topkAndUnsketched(self, hhcoords):
    #     # directly send whatever wasn't sketched
    #     unsketched = self.diff_vec[~self.sketch_mask]
    #     # COMMUNICATE
    #     return self.diff_vec[hhcoords], unsketched

    def apply_update(self, weight_update):
        weight_update = weight_update.to(self.device)
        start = 0
        for param_group in self.opt.param_groups:
            for p in param_group['params']:
                end = start + torch.numel(p)
                # we previously had diff_vec = copy - (copy - grad) = grad, so subtract here 
                p.data -= weight_update[start:end].reshape(p.data.shape)
                start = end

    def _sketcher_init(self, 
                 k=0, p2=0, numCols=0, numRows=0, numBlocks=1):
        #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids())) 
        #print(ray.get_gpu_ids())
        # set the sketched params as instance vars
        self.p2 = p2
        self.k = k
        # initialize sketch_mask, sketch, momentum buffer and accumulated grads
        # this is D
        grad_size = 0
        sketch_mask = []
        for group in self.opt.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    size = torch.numel(p)
                    if p.do_sketching:
                        sketch_mask.append(torch.ones(size))
                    else:
                        sketch_mask.append(torch.zeros(size))
                    grad_size += size
        self.grad_size = grad_size
        print(f"Total dimension is {self.grad_size} using k {self.k} and p2 {self.p2}")
        self.sketch_mask = torch.cat(sketch_mask).byte().to(self.device)
        #print(f"sketch_mask.sum(): {self.sketch_mask.sum()}")
#         print(f"Make CSVec of dim{numRows}, {numCols}")
        self.sketch = CSVec(d=self.sketch_mask.sum().item(), c=numCols, r=numRows, 
                            device=self.device, nChunks=1, numBlocks=numBlocks)


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
#args.batch_size = math.ceil(args.batch_size/args.num_workers) * args.num_workers
args.batch_size = int(args.batch_size/args.num_workers)

lr_schedule = PiecewiseLinear([0, 10, args.epochs], [0, 0.4, 0])
lr = lambda step: lr_schedule(step * args.num_workers/num_batches)/args.batch_size
print('Starting timer')
timer = Timer()

print('Preprocessing training data')
loaders = get_data_loaders(hp_default, verbose=False)
print('Finished in {:.2f} seconds'.format(timer()))

TSV = TSVLogger()

kwargs = {
    "k": args.k,
    "p2": args.p2,
    "numCols": args.cols,
    "numRows": args.rows,
    "numBlocks": args.num_blocks,
    "lr": lr,
    "num_workers": args.num_workers,
    "momentum": args.momentum,
    "optimizer" : args.optimizer,
    "criterion": args.criterion,
    "weight_decay": 5e-4*args.batch_size/args.num_workers,
    "nesterov": args.nesterov,
    "dampening": 0,
    "metrics": ['loss', 'acc'],
    "batch_size": args.batch_size,
}
ray.init(num_gpus=8)
workers = [FedSketchWorker.remote(loaders[worker_index], worker_index, kwargs) for worker_index in range(args.num_workers)]
#ps = SketchFedServer.remote(kwargs)
ps = "bleh"
def train_worker(ps, workers, iters_per_epoch, epochs, iterations, loggers=(), timer=None):
    timer = timer or Timer()
    step_number = 0
    for epoch in range(epochs):
        train_losses, train_accs, train_time = 0.0, 0.0, 0.0
        for _ in range(iters_per_epoch):
            train_loss, train_acc, step_number, diff_vecs = list(zip(*ray.get([worker.train_iters.remote(step_number, True, iterations) for worker in workers])))
        # train_loss, train_acc, step_number, diff_vecs = list(zip(*ray.get([worker.train_epoch.remote(step_number, True) for worker in workers])))
            step_number = step_number[0]
        # if epoch < 3:
            # update_vec = ps.sim_update.remote(*diff_vecs)
            #update_vec = ps.average_grads.remote(*diff_vecs)
        # else:
        # update_vec = ps.sim_update.remote(*diff_vecs)
            ray.wait([worker.all_reduce_sketched.remote(*diff_vecs) for worker in workers])
        # update_vec = torch.mean(torch.stack(diff_vecs), dim=0)
        # update_vec = ps.average_grads.remote(*diff_vecs)
        # ray.wait([worker.apply_update.remote(update_vec) for i,worker in enumerate(workers)])
            train_time += timer()
            train_losses += np.mean(np.stack([i for i in train_loss]))
            train_accs += np.mean(np.stack([i for i in train_acc]))
        test_loss, test_acc = list(zip(*ray.get([worker.train_epoch.remote(step_number, False) for worker in workers])))
        test_time = timer()
        #import pdb; pdb.set_trace()
        stats = {
            'train_time': train_time,
            'train_loss': train_losses,
            'train_acc': train_accs, 
            'test_time': test_time,
            'test_loss': np.mean(np.stack([i for i in test_loss])),
            'test_acc': np.mean(np.stack([i for i in test_acc])),
            'total_time': timer.total_time
        }
        param_values = ray.get(workers[0].fetch_opt_params.remote())
        lr = param_values['lr'] * args.batch_size
        momentum = param_values['momentum']
        weight_decay = param_values['weight_decay']
        nesterov = param_values['nesterov']
        dampening = param_values['dampening']
        #summary = union({'epoch': epoch+1, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay, 'nesterov': nesterov, 'dampening': dampening}, stats)
        #lr = param_values(step_number)['lr'] * args.batch_size
        summary = union({'epoch': epoch+1, 'lr': lr}, stats)
        for logger in loggers:
            logger.append(summary)
    return summary
import math
#updates_per_epoch = math.ceil(num_batches/(args.num_workers * args.iterations))
updates_per_epoch = int(num_batches/(args.num_workers * args.iterations))
print(f"Running {updates_per_epoch} updates for {args.iterations} iterations for {args.epochs} epochs over {num_batches} batches with {args.num_workers}")
train_worker(ps, workers, updates_per_epoch, args.epochs, args.iterations,
      loggers=(TableLogger(), TSV), timer=timer)

