import torchvision
import torch
import os
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
DATA_PATH = 'sample_data'
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