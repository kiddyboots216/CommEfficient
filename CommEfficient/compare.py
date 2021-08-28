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
from data_utils import FedSampler, FedCIFAR10, FedImageNet, FedCIFAR100, FedFEMNIST
from data_utils import cifar10_train_transforms, cifar10_test_transforms
from data_utils import cifar100_train_transforms, cifar100_test_transforms
from data_utils import imagenet_train_transforms, imagenet_val_transforms
from data_utils import femnist_train_transforms, femnist_test_transforms

import torch.multiprocessing as multiprocessing

from utils import get_param_vec

args = parse_args()
# instantiate ALL the things
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
num_classes = num_classes_of_dataset(args.dataset_name)
num_new_classes = None

model_config.update({"num_classes": num_classes,
                     "new_num_classes": num_new_classes})
model_config.update({"bn_bias_freeze": args.do_finetune,
                     "bn_weight_freeze": args.do_finetune})
if args.dataset_name == "FEMNIST":
    model_config["initial_channels"] = 1
model_cls = getattr(models, args.model)
model_mal = model_cls(**model_config)
model_ben = model_cls(**model_config)


#PATH_mal = args.checkpoint_path + args.model + str(args.mode) + str(args.do_dp) + str(True) + '.pt'
PATH_ben = args.checkpoint_path + args.model + str(args.dataset_name) + "uncompressed" + str(False) + '.pt'
#PATH_ben = args.checkpoint_path + args.model + args.finetuned_from + args.dataset_name + str(args.mode) + str(args.do_dp) + '.pt'
PATH_mal = args.checkpoint_path + args.model + str(args.dataset_name) + str(args.mode) + str(args.robustagg) + '.pt'
print("Mal Model checkpointed at ", PATH_mal)
print("Ben Model checkpointed at ", PATH_ben)
loaded_mal = torch.load(PATH_mal)
model_mal.load_state_dict(loaded_mal)
loaded_ben = torch.load(PATH_ben)
model_ben.load_state_dict(loaded_ben)
param_vec_mal = get_param_vec(model_mal)
param_vec_ben = get_param_vec(model_ben)
dist = param_vec_mal - param_vec_ben
dist_l1 = torch.norm(dist, p=1)
print(f"L1 distance is {dist_l1}")
dist_l2 = torch.norm(dist, p=2)
print(f"L2 distance is {dist_l2}")
