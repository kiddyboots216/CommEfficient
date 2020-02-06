import os
import argparse
import time
import torch
from datetime import datetime
import ctypes
import numpy as np
from collections import namedtuple
import torchvision
from collections import namedtuple

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

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

class Exp(namedtuple("Exp", ("warmup_epochs", "amplitude", "decay_len"))):
    def __call__(self, t):
        if t < self.warmup_epochs:
            return np.interp([t], [0, self.warmup_epochs], [0, self.amplitude])[0]
        else:
            return self.amplitude * 10**(-(t - self.warmup_epochs) / self.decay_len)

fed_datasets = {"CIFAR10": 10,
                "CIFAR100": 100,
                "EMNIST": 62,
                "ImageNet": 1000,
                "PERSONA": -1}

def num_classes_of_dataset(dataset_name):
    return fed_datasets[dataset_name]

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def make_logdir(args: dict):
    rows = args.num_rows
    cols = args.num_cols
    k = args.k
    mode = args.mode
    sketch_str = f"{mode}: {rows} x {cols}" if mode == "sketch" else f"{mode}"
    k_str = f"k: {k}" if mode in ["sketch", "true_topk", "local_topk"] else ""
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
    modes = ["sketch", "true_topk", "local_topk", "fedavg", "uncompressed"]
    parser.add_argument("--mode", choices=modes, default="sketch")
    parser.add_argument("--tensorboard", dest="use_tensorboard",
                        action="store_true")
    parser.add_argument("--seed", type=int, default=21)

    # data/model args
    model_names = [m for m in dir(models)
                     if m[:2] != "__" and m[0].isupper()]
    parser.add_argument("--model", default="ResNet9",
                        help="Name of the model.",
                        choices=model_names)
    parser.add_argument("--finetune", action="store_true", dest="do_finetune")
    parser.add_argument("--checkpoint", action="store_true", dest="do_checkpoint")
    parser.add_argument("--checkpoint_path", type=str,
                        default='./checkpoint',
                        help="Path or url to cache the model")
    parser.add_argument("--finetune_path", type=str,
                        default='./finetune',
                        help="Path or url of the model cache")
    parser.add_argument("--finetuned_from", type=str,
                        help="Name of the dataset you pretrained on.",
                        choices=fed_datasets.keys())
    parser.add_argument("--num_results_train", type=int, default=2)
    parser.add_argument("--num_results_val", type=int, default=2)
    parser.add_argument("--dataset_name", type=str, default="",
                        help="Name of the dataset.",
                        choices=fed_datasets.keys())
    parser.add_argument("--dataset_dir", type=str,
                        default='./dataset',
                        help="Path or url of the dataset cache")
    parser.add_argument("--batchnorm", action="store_true", dest="do_batchnorm")
    parser.add_argument("--nan_threshold", type=float, default=999)

    # compression args
    parser.add_argument("--k", type=int, default=50000)
    parser.add_argument("--num_cols", type=int, default=500000)
    parser.add_argument("--num_rows", type=int, default=5)
    parser.add_argument("--num_blocks", type=int, default=20)
    parser.add_argument("--topk_down", action="store_true",
                        dest="do_topk_down")

    # optimization args
    parser.add_argument("--local_momentum", type=float, default=0.9)
    parser.add_argument("--virtual_momentum", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=float, default=24,
                        help="Number of training epochs")
    parser.add_argument("--num_fedavg_epochs", type=int, default=1)
    parser.add_argument("--fedavg_batch_size", type=int, default=-1)
    parser.add_argument("--fedavg_lr_decay", type=float, default=1)
    error_types = ["none", "local", "virtual"]
    parser.add_argument("--error_type", choices=error_types,
                        default="none")
    parser.add_argument("--lr_scale", type=float, default=default_lr)
    parser.add_argument("--pivot_epoch", type=float, default=5)

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
    parser.add_argument("--share_ps_gpu", action="store_true")
    parser.add_argument("--iid", action="store_true", dest="do_iid")
    parser.add_argument("--train_dataloader_workers",
                        type=int, default=0)
    parser.add_argument("--val_dataloader_workers",
                        type=int, default=0)

    # GPT2 args
    parser.add_argument("--model_checkpoint", type=str, default="gpt2",
                        help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2,
                        help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2,
                        help=("Number of previous exchanges to keep"
                              " in history"))
    parser.add_argument("--local_batch_size", type=int, default=8,
                        help="Batch size for training (-1 uses all data the client has)")
    parser.add_argument("--valid_batch_size", type=int, default=8,
                        help="Batch size for validation")
    parser.add_argument("--microbatch_size", type=int, default=-1,
                        help=("Size of each batch shard to be processed to save memory (-1 uses all data)"))
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

    # Differential Privacy args
    parser.add_argument("--dp", action="store_true", dest="do_dp", help=("Whether to do differentially private training)"))
    dp_modes = ["worker", "server"]
    parser.add_argument("--dp_mode", choices=dp_modes, default="worker")
    parser.add_argument("--l2_norm_clip", type=float, default=1.0, help=("What value to clip the l2 norm to"))
    parser.add_argument("--noise_multiplier", type=float, default=0.0, help=("Sigma, i.e. standard dev of noise"))

    args = parser.parse_args()
    port_in_use = True
    while port_in_use:
        if is_port_in_use(args.port):
            print(f"{args.port} port in use, trying next...")
            args.port += np.random.randint(0,1000)
        else:
            port_in_use = False

    if args.mode == "fedavg":
        assert args.local_batch_size == -1
        assert args.local_momentum == 0
        assert args.error_type == "none"

    return args

def _topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    # on a gpu, sorting is faster than pytorch's topk method
    #topkIndices = torch.sort(vec**2)[1][-k:]
    # however, torch.topk is more space efficient

    # topk on cuda returns what looks like uninitialized memory if
    # vals has nan values in it
    # saving to a zero-initialized output array instead of using the
    # output of topk appears to solve this problem
    topkVals = torch.zeros(k, device=vec.device)
    topkIndices = torch.zeros(k, device=vec.device).long()
    torch.topk(vec**2, k, sorted=False, out=(topkVals, topkIndices))

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
            if p.requires_grad:
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
    param_vec = []
    for p in model.parameters():
        if p.requires_grad:
            param_vec.append(p.data.view(-1).float())
    return torch.cat(param_vec)

    #return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_param_vec(model, param_vec):
    start = 0
    for p in model.parameters():
        if p.requires_grad:
            end = start + p.numel()
            p.data.zero_()
            p.data.add_(param_vec[start:end].view(p.size()))
            start = end

def sm2np(sm, shape, dtype=ctypes.c_float):
    # convert from shared memory object/buffer to numpy array
    nparray = np.ndarray(shape, dtype=dtype, buffer=sm)
    assert(nparray.base is sm)
    return nparray

def clip_grad(l2_norm_clip, record):
    try:
        l2_norm = torch.norm(record)
    except:
        l2_norm = record.l2estimate()
    if l2_norm < l2_norm_clip:
        return record
    else:
        return record / float(torch.abs(torch.tensor(l2_norm) / l2_norm_clip))

def steps_per_epoch(local_batch_size, dataset, num_workers):
    if local_batch_size == -1:
        spe = dataset.num_clients // num_workers
    else:
        batch_size = local_batch_size * num_workers
        spe = np.ceil(len(dataset) / batch_size)
    return spe

