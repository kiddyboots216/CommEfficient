import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import namedtuple
import os
from torch.utils.data.dataset import Dataset

#####################
## data preprocessing
#####################

# equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_mean = (0.4914, 0.4822, 0.4465)
# equals np.std(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)],
                  mode='reflect')

#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)

class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True,
                                             download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True)
    return {
        'train': {'data': train_set.train_data,
                  'labels': train_set.train_labels},
        'test': {'data': test_set.test_data,
                 'labels': test_set.test_labels}
    }

class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        assert self.choices is not None
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            if hasattr(t, 'output_shape'):
                x_shape = t.output_shape(x_shape)
            self.choices.append({k:np.random.choice(v, size=N)
                                 for (k,v) in options.items()})

#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle,
                 set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ([x, y.long()] for (x,y) in self.dataloader)
        return ({'input': x, 'target': y.long()}
                for (x,y) in self.dataloader)
        return ({'input': x.cuda(), 'target': y.cuda().long()}
                for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


DATA_PATH = 'sample_data'

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

def get_cifar10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = torchvision.datasets.CIFAR10(
            root=os.path.join(DATA_PATH, "CIFAR10"),
            train=True,
            download=True
        )
    data_test = torchvision.datasets.CIFAR10(
            root=os.path.join(DATA_PATH, "CIFAR10"),
            train=False,
            download=True
        )

    x_train = data_train.train_data
    y_train = np.array(data_train.train_labels)
    x_test = data_test.test_data
    y_test = np.array(data_test.test_labels)

    return x_train, y_train, x_test, y_test

    x_train = data_train.train_data.transpose((0,3,1,2))
    y_train = np.array(data_train.train_labels)
    x_test = data_test.test_data.transpose((0,3,1,2))
    y_test = np.array(data_test.test_labels)

    return x_train, y_train, x_test, y_test

def split_image_data(data, labels, n_clients=10, classes_per_client=10,
                     shuffle=True, verbose=True, balancedness=None):
    '''
    Splits (data, labels) evenly among 'n_clients s.t.
        every client holds 'classes_per_client different labels
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    '''
    # constants
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    if balancedness >= 1.0:
        data_per_client = [n_data // n_clients] * n_clients
        n_per_client_per_class = data_per_client[0] // classes_per_client
        data_per_client_per_class = [n_per_client_per_class] * n_clients
    else:
        fracs = balancedness**np.linspace(0,n_clients-1, n_clients)
        fracs /= np.sum(fracs)
        fracs = 0.1/n_clients + (1-0.1)*fracs
        data_per_client = [np.floor(frac * n_data).astype('int')
                           for frac in fracs]

        data_per_client = data_per_client[::-1]

        data_per_client_per_class = [np.maximum(1,nd // classes_per_client)
                                     for nd in data_per_client]

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
            take = min(data_per_client_per_class[i],
                       len(data_idcs[c]),
                       budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            label_range = np.arange(n_labels).reshape(-1,1)
            split = np.sum(client[1].reshape(1, -1) == label_range, axis=1)
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
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))]),
                                    #(0.24703223, 0.24348513, 0.26158784)
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
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))]),
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

    transforms_train, transforms_eval = get_default_data_transforms(
            hp['dataset'], verbose=False)

    split = split_image_data(x_train, y_train, n_clients=hp['n_clients'],
                             classes_per_client=hp['classes_per_client'],
                             balancedness=hp['balancedness'],
                             verbose=verbose)
    datasets = [CustomImageDataset(x, y, transforms_train)
                for x, y in split]
    client_loaders = [torch.utils.data.DataLoader(
                            dataset, batch_size=hp['batch_size']
                        )
                      for dataset in datasets]
    #import pdb; pdb.set_trace()
    #client_loaders = [torch.utils.data.DataLoader(
    #                       CustomImageDataset(x, y, transforms_train),
    #                       batch_size=hp['batch_size'],
    #                       shuffle=True)
    #                  for x, y in split]
    train_loader = torch.utils.data.DataLoader(
            CustomImageDataset(x_train, y_train, transforms_eval),
            batch_size=512,
            shuffle=False
        )
    test_loader  = torch.utils.data.DataLoader(
            CustomImageDataset(x_test, y_test, transforms_eval),
            batch_size=512,
            shuffle=False
        )

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
