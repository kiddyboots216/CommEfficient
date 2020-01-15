import os
import argparse
import time
import torch
from datetime import datetime
import ctypes
import numpy as np
from collections import namedtuple
import torchvision

import models

class Logger:
    def debug(self, msg, args=None):
        print(msg.format(args))
    def info(self, msg, args=None):
        print(msg.format(args))
    def warn(self, msg, args=None):
        print(msg.format(args))
    def error(self, msg, args=None):
        print(msg.format(args))
    def critical(self, msg, args=None):
        print(msg.format(args))

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def make_logdir(args: dict):
    rows = args.num_rows
    cols = args.num_cols
    k = args.k
    mode = args.mode
    num_local_iters = args.num_local_iters
    sketch_str = f"{mode}: {rows} x {cols}" if mode == "sketch" else f"{mode}"
    k_str = f"k: {k}" if mode in ["sketch", "true_topk", "local_topk"] else f"num_local_iters: {num_local_iters}"
    workers = args.num_workers
    clients = args.num_clients
    clients_str = f"{workers}/{clients}"
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + clients_str + '_' + sketch_str + '_' + k_str)
    return logdir

class TableLogger():
    def append(self, output):
        if not hasattr(self, 'keys'):
            self.keys = output.keys()
            print(*('{:>12s}'.format(k) for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(*('{:12.4f}'.format(v)
                 if isinstance(v, np.float) or isinstance(v, np.float32) else '{:12}'.format(v)
                for v in filtered))

class TSVLogger():
    def __init__(self):
        self.log = ['epoch,hours,top1Accuracy']
    def append(self, output):
        epoch = output['epoch']
        hours = output['total_time']/3600
        acc = output['test_acc']*100
        self.log.append('{},{:.8f},{:.2f}'.format(epoch, hours, acc))
    def __str__(self):
        return '\n'.join(self.log)

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

class Timer():
    def __init__(self):
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t


def parse_args(default_lr=None):
    parser = argparse.ArgumentParser()

    # meta-args
    parser.add_argument("--test", action="store_true", dest="do_test")
    modes = ["sketch", "true_topk", "local_topk", "localSGD", "uncompressed"]
    parser.add_argument("--mode", choices=modes, default="sketch")
    parser.add_argument("--tensorboard", dest="use_tensorboard",
                        action="store_true")

    # data/model args
    parser.add_argument("--num_data", type=int, default=50000)
    model_names = [m for m in dir(models)
                     if m[:2] != "__" and m[0].isupper()]
    parser.add_argument("--model", default="ResNet9",
                        help="Name of the model.",
                        choices=model_names)
    parser.add_argument("--num_results_train", type=int, default=2)
    parser.add_argument("--num_results_val", type=int, default=2)
    parser.add_argument("--supervised", action="store_true",
                        dest="is_supervised")
    fed_datasets = ["CIFAR10", "ImageNet"]
    parser.add_argument("--dataset_name", type=str, default="",
                        help="Name of the dataset.",
                        choices=fed_datasets)
    parser.add_argument("--dataset_dir", type=str,
                        default='./dataset',
                        help="Path or url of the dataset cache")

    # compression args
    parser.add_argument("--k", type=int, default=50000)
    parser.add_argument("--num_cols", type=int, default=500000)
    parser.add_argument("--num_rows", type=int, default=5)
    parser.add_argument("--num_blocks", type=int, default=20)
    parser.add_argument("--topk_down", action="store_true",
                        dest="do_topk_down")

    # optimization args
    parser.add_argument("--nesterov", action="store_true",
                        dest="do_nesterov")
    parser.add_argument("--local_momentum", type=float, default=0.9)
    parser.add_argument("--virtual_momentum", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=24,
                        help="Number of training epochs")
    momentum_types = ["none", "local", "virtual"]
    parser.add_argument("--momentum_type", choices=momentum_types,
                        default="none")
    error_types = momentum_types
    parser.add_argument("--error_type", choices=error_types,
                        default="none")
    reductions = ["mean", "median"]
    parser.add_argument("--grad_reduction",
                        choices=reductions,
                        default="mean",
                        help="How to combine gradients from workers")
    parser.add_argument("--lr_scale", type=float, default=default_lr)
    parser.add_argument("--pivot_epoch", type=int, default=5)
    parser.add_argument("--mixup_alpha", type=float, default=1)
    parser.add_argument("--mixup", action="store_true", dest="do_mixup")

    # parallelization args
    parser.add_argument("--port", type=int, default=5315)
    parser.add_argument("--num_clients", type=int)
    parser.add_argument("--num_workers", type=int, default=1)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                        default=default_device,
                        help="Device (cuda or cpu)")
    parser.add_argument("--num_devices", type=int,
                        default=1,
                        help="Number of gpus")
    parser.add_argument("--num_local_iters", type=int, default=1)
    parser.add_argument("--local_sched", action="store_true", dest="use_local_sched")
    parser.add_argument("--share_ps_gpu", action="store_true")
    parser.add_argument("--iid", action="store_true", dest="do_iid")

    # GPT2 args
    parser.add_argument("--num_dialogs", type=int, default=1)
    parser.add_argument("--model_checkpoint", type=str, default="gpt2",
                        help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2,
                        help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2,
                        help=("Number of previous exchanges to keep"
                              " in history"))
    parser.add_argument("--local_batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8,
                        help="Batch size for validation")
    parser.add_argument("--num_train_batch_shards", type=int,
                        default=1,
                        help=("Split up each batch into shards"
                              " to save memory"))
    parser.add_argument("--num_val_batch_shards", type=int,
                        default=1,
                        help=("Split up each batch into shards"
                              " to save memory"))
    parser.add_argument("--lm_coef", type=float, default=1.0,
                        help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0,
                        help="Multiple-choice loss coefficient")
    parser.add_argument("--max_grad_norm", type=float,
                        help="Clipping gradient norm, is per-worker")
    parser.add_argument("--personality_permutations", type=int, default=1,
                        help=("Number of permutations of personality"
                              " sentences"))
    parser.add_argument("--eval_before_start", action='store_true',
                        help=("If true start with a first evaluation"
                              " before training"))
    parser.add_argument("--fp16", type=str, default="",
                        help=("Set to O0, O1, O2 or O3 for fp16 training"
                              " (see apex documentation)"))


    args = parser.parse_args()
    port_in_use = True
    while port_in_use:
        if is_port_in_use(args.port):
            print(f"{args.port} port in use, trying next...")
            args.port += np.random.randint(0,1000)
        else:
            port_in_use = False

    return args

def _topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    # on a gpu, sorting is faster than pytorch's topk method
    #topkIndices = torch.sort(vec**2)[1][-k:]
    # however, torch.topk is more space efficient
    topkIndices = torch.topk(vec**2, k, sorted=False)[1]

    ret = torch.zeros_like(vec)
    if len(vec.size()) == 1:
        ret[topkIndices] = vec[topkIndices]
    elif len(vec.size()) == 2:
        rows = torch.arange(vec.size()[0]).view(-1,1)
        ret[rows, topkIndices] = vec[rows, topkIndices]
    return ret

def get_grad(model, args):
    weights = get_param_vec(model)
    grad_vec = get_grad_vec(model)
    if args.weight_decay != 0:
        grad_vec.add_(args.weight_decay / args.num_workers, weights)
    return grad_vec.to(args.device)

def get_grad_vec(model):
    grad_vec = []
    with torch.no_grad():
        # flatten
        for p in model.parameters():
            if p.grad is None:
                grad_vec.append(torch.zeros_like(p.data.view(-1)))
            else:
                grad_vec.append(p.grad.data.view(-1).float())
        # concat into a single vector
        grad_vec = torch.cat(grad_vec)
    return grad_vec

def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

def get_param_vec(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_param_vec(model, param_vec):
    start = 0
    for p in model.parameters():
        end = start + p.numel()
        p.data.zero_()
        p.data.add_(param_vec[start:end].view(p.size()))
        start = end

def sm2np(sm, shape, dtype=ctypes.c_float):
    # convert from shared memory object/buffer to numpy array
    nparray = np.ndarray(shape, dtype=dtype, buffer=sm)
    assert(nparray.base is sm)
    return nparray
