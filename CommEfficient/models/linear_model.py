__all__ = ["LinearModel"]

import torch
import torch.nn as nn
import numpy as np

def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x

class StandardizeLayer(nn.Module):
    def __init__(self, bn_stats):
        super(StandardizeLayer, self).__init__()
        self.bn_stats = bn_stats

    def forward(self, x):
        return standardize(x, self.bn_stats)

class LinearModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        dataset_dir = "/data/scsi/ashwineep/datasets/CIFAR10Pretrained/"
        feature_path = dataset_dir + "features/cifar100_resnext"
        x_train = np.load(f"{feature_path}_train.npy")
        n_features = x_train.shape[-1]
        mean = np.load(f"{feature_path}_mean.npy")
        var = np.load(f"{feature_path}_var.npy")
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        bn_stats = (torch.from_numpy(mean).to(device), torch.from_numpy(var).to(device))

        self.model = nn.Sequential(StandardizeLayer(bn_stats), nn.Linear(n_features, 10))

    def forward(self, x):
        x = x.reshape((-1, 1024))
        return self.model(x)
