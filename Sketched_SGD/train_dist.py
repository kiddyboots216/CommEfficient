import os
import collections
import logging
import glob
import re
from functools import singledispatch
from collections import OrderedDict
from collections import namedtuple
from inspect import signature

import torch, torchvision
import numpy as np
import ray

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset

import itertools as it
import copy

LARGEPRIME = 2**61-1

cache = {}

#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

class CSVec(object):
    """ Simple Count Sketched Vector """
    def __init__(self, d, c, r, doInitialize=True, device=None,
                 nChunks=1, numBlocks=1):
        global cache

        self.r = r # num of rows
        self.c = c # num of columns
        # need int() here b/c annoying np returning np.int64...
        self.d = int(d) # vector dimensionality
        # how much to chunk up (on the GPU) any computation
        # that requires computing something along all tokens. Doing
        # so saves GPU RAM at the cost of having to transfer the chunks
        # of self.buckets and self.signs between host & device
        self.nChunks = nChunks

        # reduce memory consumption of signs & buckets by constraining
        # them to be repetitions of a single block
        self.numBlocks = numBlocks

#         if device is None:
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         else:
#         assert("cuda" in device or device == "cpu")
        self.device = device
#         print(f"CSVec is using backend {self.device}")

        if not doInitialize:
            return

        # initialize the sketch
        self.table = torch.zeros((self.r, self.c), device=self.device)
#         print(f"Making table of dim{self.r,self.c} which is {self.table}")

        # if we already have these, don't do the same computation
        # again (wasting memory storing the same data several times)
        if (d, c, r) in cache:
            hashes = cache[(d, c, r)]["hashes"]
            self.signs = cache[(d, c, r)]["signs"]
            self.buckets = cache[(d, c, r)]["buckets"]
            if self.numBlocks > 1:
                self.blockSigns = cache[(d, c, r)]["blockSigns"]
                self.blockOffsets = cache[(d, c, r)]["blockOffsets"]
            return

        # initialize hashing functions for each row:
        # 2 random numbers for bucket hashes + 4 random numbers for
        # sign hashes
        # maintain existing random state so we don't mess with
        # the main module trying to set the random seed but still
        # get reproducible hashes for the same value of r

        # do all these computations on the CPU, since pytorch
        # is incapable of in-place mod, and without that, this
        # computation uses up too much GPU RAM
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        hashes = torch.randint(0, LARGEPRIME, (r, 6),
                                    dtype=torch.int64, device="cpu")

        if self.numBlocks > 1:
            nTokens = self.d // numBlocks
            if self.d % numBlocks != 0:
                # so that we only need numBlocks repetitions
                nTokens += 1
            self.blockSigns = torch.randint(0, 2, size=(self.numBlocks,),
                                            device=self.device) * 2 - 1
            self.blockOffsets = torch.randint(0, self.c,
                                              size=(self.numBlocks,),
                                              device=self.device)
        else:
            assert(numBlocks == 1)
            nTokens = self.d

        torch.random.set_rng_state(rand_state)

        tokens = torch.arange(nTokens, dtype=torch.int64, device="cpu")
        tokens = tokens.reshape((1, nTokens))

        # computing sign hashes (4 wise independence)
        h1 = hashes[:,2:3]
        h2 = hashes[:,3:4]
        h3 = hashes[:,4:5]
        h4 = hashes[:,5:6]
        self.signs = (((h1 * tokens + h2) * tokens + h3) * tokens + h4)
        self.signs = ((self.signs % LARGEPRIME % 2) * 2 - 1).float()
        if self.nChunks == 1:
            # only move to device now, since this computation takes too
            # much memory if done on the GPU, and it can't be done
            # in-place because pytorch (1.0.1) has no in-place modulo
            # function that works on large numbers
            self.signs = self.signs.to(self.device)

        # computing bucket hashes  (2-wise independence)
        h1 = hashes[:,0:1]
        h2 = hashes[:,1:2]
        self.buckets = ((h1 * tokens) + h2) % LARGEPRIME % self.c
        if self.nChunks == 1:
            # only move to device now. See comment above.
            # can't cast this to int, unfortunately, since we index with
            # this below, and pytorch only lets us index with long
            # tensors
            self.buckets = self.buckets.to(self.device)

        cache[(d, c, r)] = {"hashes": hashes,
                            "signs": self.signs,
                            "buckets": self.buckets}
        if numBlocks > 1:
            cache[(d, c, r)].update({"blockSigns": self.blockSigns,
                                     "blockOffsets": self.blockOffsets})

    def zero(self):
        self.table.zero_()

    def __deepcopy__(self, memodict={}):
        # don't initialize new CSVec, since that will calculate bc,
        # which is slow, even though we can just copy it over
        # directly without recomputing it
        newCSVec = CSVec(d=self.d, c=self.c, r=self.r,
                         doInitialize=False, device=self.device,
                         nChunks=self.nChunks, numBlocks=self.numBlocks)
        newCSVec.table = copy.deepcopy(self.table)
        global cache
        cachedVals = cache[(self.d, self.c, self.r)]
        newCSVec.hashes = cachedVals["hashes"]
        newCSVec.signs = cachedVals["signs"]
        newCSVec.buckets = cachedVals["buckets"]
        if self.numBlocks > 1:
            newCSVec.blockSigns = cachedVals["blockSigns"]
            newCSVec.blockOffsets = cachedVals["blockOffsets"]
        return newCSVec

    def __add__(self, other):
        # a bit roundabout in order to avoid initializing a new CSVec
        returnCSVec = copy.deepcopy(self)
        returnCSVec += other
        return returnCSVec

    def __iadd__(self, other):
        if isinstance(other, CSVec):
            self.accumulateCSVec(other)
        elif isinstance(other, torch.Tensor):
#             self.accumulateTable(other)
            self.accumulateVec(other)
        else:
#             from IPython.core.debugger import set_trace; set_trace()
            raise ValueError(f"Can't add this to a CSVec: {other} because it is not a {CSVec}")
        return self

    def accumulateVec(self, vec):
        # updating the sketch
        try:
            assert(len(vec.size()) == 1 and vec.size()[0] == self.d), f"Len of {vec} was {len(vec.size())} instead of 1 or size was {vec.size()[0]} instead of {self.d}"
        except AssertionError:
            return self.accumulateTable(vec)
#             vec = torch.squeeze(vec, 0)
#             assert(len(vec.size()) == 1 and vec.size()[0] == self.d), f"After squeeze, Len was {len(vec.size())} instead of 1 or size was {vec.size()[0]} instead of {self.d}"
        for r in range(self.r):
            buckets = self.buckets[r,:].to(self.device)
            signs = self.signs[r,:].to(self.device)
            for blockId in range(self.numBlocks):
                start = blockId * buckets.size()[0]
                end = (blockId + 1) * buckets.size()[0]
                end = min(end, self.d)
                offsetBuckets = buckets[:end-start].clone()
                offsetSigns = signs[:end-start].clone()
                if self.numBlocks > 1:
                    offsetBuckets += self.blockOffsets[blockId]
                    offsetBuckets %= self.c
                    offsetSigns *= self.blockSigns[blockId]
                self.table[r,:] += torch.bincount(
                                    input=offsetBuckets,
                                    weights=offsetSigns * vec[start:end],
                                    minlength=self.c
                                   )
                #self.table[r,:] += torch.ones(self.c).to(self.device)

        """
        for i in range(self.nChunks):
            start = int(i / self.nChunks * self.d)
            end = int((i + 1) / self.nChunks * self.d)
            # this will be idempotent if nChunks == 1
            buckets = self.buckets[:,start:end].to(self.device)
            signs = self.signs[:,start:end].to(self.device)
            for r in range(self.r):
                self.table[r,:] += torch.bincount(
                                    input=buckets[r,:],
                                    weights=signs[r,:] * vec[start:end],
                                    minlength=self.c
                                   )
                #self.table[r,:] += torch.ones(self.c)
                #pass
        """
    def accumulateTable(self, table):
        assert self.table.size() == table.size(), f"This CSVec is {self.table.size()} but the table is {table.size()}"
        self.table += table
    
    def accumulateCSVec(self, csVec):
        # merges csh sketch into self
        assert(self.d == csVec.d)
        assert(self.c == csVec.c)
        assert(self.r == csVec.r)
        self.table += csVec.table

    #@profile
    def _findHHK(self, k):
        #return torch.arange(k).to(self.device), torch.arange(k).to(self.device).float()
        assert(k is not None)
        #tokens = torch.arange(self.d, device=self.device)
        #vals = self._findValues(tokens)
        vals = self._findAllValues()
        #vals = torch.arange(self.d).to(self.device).float()
        # sort is faster than torch.topk...
        HHs = torch.sort(vals**2)[1][-k:]
        #HHs = torch.topk(vals**2, k, sorted=False)[1]
        return HHs, vals[HHs]

    def _findHHThr(self, thr):
        assert(thr is not None)
        # to figure out which items are heavy hitters, check whether
        # self.table exceeds thr (in magnitude) in at least r/2 of
        # the rows. These elements are exactly those for which the median
        # exceeds thr, but computing the median is expensive, so only
        # calculate it after we identify which ones are heavy
        tablefiltered = (  (self.table >  thr).float()
                         - (self.table < -thr).float())
        est = torch.zeros(self.d, device=self.device)
        for r in range(self.r):
            est += tablefiltered[r,self.buckets[r,:]] * self.signs[r,:]
        est = (  (est >=  math.ceil(self.r/2.)).float()
               - (est <= -math.ceil(self.r/2.)).float())

        # HHs - heavy coordinates
        HHs = torch.nonzero(est)
        return HHs, self._findValues(HHs)

    def _findValues(self, coords):
        # estimating frequency of input coordinates
        assert(self.numBlocks == 1)
        chunks = []
        d = coords.size()[0]
        if self.nChunks == 1:
            vals = torch.zeros(self.r, self.d, device=self.device)
            for r in range(self.r):
                vals[r] = (self.table[r, self.buckets[r, coords]]
                           * self.signs[r, coords])
            return vals.median(dim=0)[0]

        # if we get here, nChunks > 1
        for i in range(self.nChunks):
            vals = torch.zeros(self.r, d // self.nChunks,
                               device=self.device)
            start = int(i / self.nChunks * d)
            end = int((i + 1) / self.nChunks * d)
            buckets = self.buckets[:,coords[start:end]].to(self.device)
            signs = self.signs[:,coords[start:end]].to(self.device)
            for r in range(self.r):
                vals[r] = self.table[r, buckets[r, :]] * signs[r, :]
            # take the median over rows in the sketch
            chunks.append(vals.median(dim=0)[0])

        vals = torch.cat(chunks, dim=0)
        return vals

    def _findAllValues(self):
#         from IPython.core.debugger import set_trace; set_trace()
        if self.nChunks == 1:
            if self.numBlocks == 1:
                vals = torch.zeros(self.r, self.d, device=self.device)
                for r in range(self.r):
                    vals[r] = (self.table[r, self.buckets[r,:]]
                               * self.signs[r,:])
#                 print(f"Table of size {self.r, self.d} is {self.table}")
                return vals.median(dim=0)[0]
            else:
                medians = torch.zeros(self.d, device=self.device)
                #ipdb.set_trace()
                for blockId in range(self.numBlocks):
                    start = blockId * self.buckets.size()[1]
                    end = (blockId + 1) * self.buckets.size()[1]
                    end = min(end, self.d)
                    vals = torch.zeros(self.r, end-start, device=self.device)
                    for r in range(self.r):
                        buckets = self.buckets[r, :end-start]
                        signs = self.signs[r, :end-start]
                        offsetBuckets = buckets + self.blockOffsets[blockId]
                        offsetBuckets %= self.c
                        offsetSigns = signs * self.blockSigns[blockId]
                        vals[r] = (self.table[r, offsetBuckets]
                                    * offsetSigns)
                    medians[start:end] = vals.median(dim=0)[0]
                return medians

    def findHHs(self, k=None, thr=None):
        assert((k is None) != (thr is None))
        if k is not None:
            return self._findHHK(k)
        else:
            return self._findHHThr(thr)

    def unSketch(self, k=None, epsilon=None):
        # either epsilon or k might be specified
        # (but not both). Act accordingly
        if epsilon is None:
            thr = None
        else:
            thr = epsilon * self.l2estimate()

        hhs = self.findHHs(k=k, thr=thr)

        if k is not None:
            assert(len(hhs[1]) == k), f"Should have found {k} hhs but only found {len(hhs[1])}"
        if epsilon is not None:
            assert((hhs[1] < thr).sum() == 0)

        # the unsketched vector is 0 everywhere except for HH
        # coordinates, which are set to the HH values
        unSketched = torch.zeros(self.d, device=self.device)
        unSketched[hhs[0]] = hhs[1]
        return unSketched

    def l2estimate(self):
        # l2 norm esimation from the sketch
        return np.sqrt(torch.median(torch.sum(self.table**2,1)).item())

from inspect import signature
from collections import namedtuple
import time
import numpy as np
import pandas as pd
from functools import singledispatch
from collections import OrderedDict
import torch
import torch.nn as nn
import track
import torch.nn.functional as F
# from single_trainer import SGD_Sketched
#####################
# utils
#####################

class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

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

localtime = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

class TableLogger():
    def append(self, output):
        if not hasattr(self, 'keys'):
            self.keys = output.keys()
            print(*('{:>12s}'.format(k) for k in self.keys))
        filtered = [output[k] for k in self.keys]
        #import pdb; pdb.set_trace()
        print(*('{:12.4f}'.format(v)
                 if isinstance(v, np.float) or isinstance(v, np.float32) else '{:12}'.format(v)
                for v in filtered))

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

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

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
## dict utils
#####################

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, (*pfx, name))
        else: yield ((*pfx, name), val)

#####################
## graph building
#####################

sep='_'
RelativePath = namedtuple('RelativePath', ('parts'))
rel_path = lambda *parts: RelativePath(parts)

def build_graph(net):
    net = OrderedDict(path_iter(net))
    default_inputs = [[('input',)]]+[[k] for k in net.keys()]
    with_default_inputs = lambda vals: (
            val if isinstance(val, tuple) else (val, default_inputs[idx])
            for idx,val in enumerate(vals)
        )
    # srsly?
    parts = lambda path, pfx: tuple(pfx) + path.parts if isinstance(path, RelativePath) else (path,) if isinstance(path, str) else path
    r = OrderedDict(
            [(sep.join((*pfx, name)), (val, [sep.join(parts(x, pfx))
                                           for x in inputs]))
             for (*pfx, name), (val, inputs) in zip(
                                         net.keys(),
                                         with_default_inputs(net.values())
                                     )
            ])
    return r

import numpy as np
import torch
from torch import nn
import torchvision

#####################
## dataset
#####################

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
        return ({'input': x.cuda(), 'target': y.cuda().long()}
                for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)

#####################
## torch stuff
#####################

class Identity(nn.Module):
    def forward(self, x): return x

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x):
        return x*self.weight

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Add(nn.Module):
    def forward(self, x, y): return x + y

class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)


def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False,
               bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False

    return m

#Network definition
class ConvBN(nn.Module):
    def __init__(self, c_in, c_out, bn_weight_init=1.0, pool=None, **kw):
        super().__init__()
        self.pool = pool
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn = batch_norm(c_out, bn_weight_init=bn_weight_init, **kw)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        if self.pool:
            out = self.pool(out)
        return out
    

class Residual(nn.Module):
    def __init__(self, c, **kw):
        super().__init__()
        self.res1 = ConvBN(c, c, **kw)
        self.res2 = ConvBN(c, c, **kw)

    def forward(self, x):
        return x + F.relu(self.res2(self.res1(x)))

class BasicNet(nn.Module):
    def __init__(self, channels, weight,  pool, **kw):
        super().__init__()
        self.prep = ConvBN(3, channels['prep'], **kw)

        self.layer1 = ConvBN(channels['prep'], channels['layer1'],
                             pool=pool, **kw)
        self.res1 = Residual(channels['layer1'], **kw)

        self.layer2 = ConvBN(channels['layer1'], channels['layer2'],
                             pool=pool, **kw)

        self.layer3 = ConvBN(channels['layer2'], channels['layer3'],
                             pool=pool, **kw)
        self.res3 = Residual(channels['layer3'], **kw)

        self.pool = nn.MaxPool2d(4)
        self.linear = nn.Linear(channels['layer3'], 10, bias=False)
        self.classifier = Mul(weight)

    def forward(self, x):
        out = self.prep(x)
        out = self.res1(self.layer1(out))
        out = self.layer2(out)
        out = self.res3(self.layer3(out))

        out = self.pool(out).view(out.size()[0], -1)
        out = self.classifier(self.linear(out))
        return out

class Net(nn.Module):
    def __init__(self, channels=None, weight=0.125, pool=nn.MaxPool2d(2),
                 extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
        super().__init__()
        channels = channels or {'prep': 64, 'layer1': 128,
                                'layer2': 256, 'layer3': 512}
        self.n = BasicNet(channels, weight, pool, **kw)
        #for layer in res_layers:
        #    n[layer]['residual'] = residual(channels[layer], **kw)
        #for layer in extra_layers:
        #    n[layer]['extra'] = ConvBN(channels[layer], channels[layer], **kw)
    def forward(self, x):
        return self.n(x)
    
class TSVLogger():
    def __init__(self):
        self.log = ['epoch\thours\ttop1Accuracy']
    def append(self, output):
        epoch = output['epoch']
        hours = output['total_time']/3600
        acc = output['test_acc']*100
        self.log.append('{}\t{:.8f}\t{:.2f}'.format(epoch, hours, acc))
    def __str__(self):
        return '\n'.join(self.log)

class SketchedModel:
    def __init__(self, model_cls, model_config, workers, 
                sketch_biases=False, sketch_params_larger_than=0):
        self.workers = workers
        self.model = model_cls(**model_config)
        [worker.set_model.remote(model_cls, model_config, 
            sketch_biases, sketch_params_larger_than) for worker in self.workers]
        #[worker.set_model(self.model) for worker in self.workers]

    def __call__(self, *args, **kwargs):
        input_minibatches = []
        batch_size = len(args[0])
        #import pdb; pdb.set_trace()
        num_workers = len(self.workers)
        for i, _ in enumerate(self.workers):
            start = i * batch_size // num_workers
            end = (i+1) * batch_size // num_workers
            input_minibatches.append(args[0][start:end])
            # target_minibatches.append(targets[start:end])
        return [worker.model_call.remote(
            input_minibatches[worker_id]) for worker_id, worker in enumerate(self.workers)]
        #return [worker.model_call(*args) for worker in self.workers]

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        if name in ["model", "workers"]:
            self.__dict__[name] = value
        else:
            [worker.model_setattr.remote(name, value) for worker in self.workers]

class SketchedLoss(object):
    def __init__(self, criterion, workers):
        self.workers = workers
        [worker.set_loss.remote(criterion) for worker in self.workers]

    def __call__(self, *args, **kwargs):
        #import pdb; pdb.set_trace()
        if len(kwargs) > 0:
            print("Kwargs aren't supported by Ray")
            return
        input_minibatches = args[0]
        target_minibatches = []
        batch_size = len(args[1])
        #assert len(args[0]) == len(args[1]), f"{len(args[0])} != {len(args[1])}"
        #import pdb; pdb.set_trace()
        num_workers = len(self.workers)
        for i, _ in enumerate(self.workers):
            start = i * batch_size // num_workers
            end = (i+1) * batch_size // num_workers
            #input_minibatches.append(args[0][start:end])
            target_minibatches.append(args[1][start:end])
            # target_minibatches.append(targets[start:end])
        # TODO: fix this partitioning
        # results = [worker.loss_call.remote(args)
        #            for worker_id, worker in enumerate(self.workers)]
#         for worker_id, worker in enumerate(self.workers):
#             worker.loss_call.remote(args[worker_id])
#         [worker.loss_call.remote()
#         for worker_id, worker in enumerate(self.workers)]
#         results = torch.zeros(2)
        results = torch.stack(
             ray.get(
                 [worker.loss_call.remote(
                    input_minibatches[worker_id], target_minibatches[worker_id])
                 for worker_id, worker in enumerate(self.workers)]
             ), 
             dim=0)
        #results.register_backward_hook(self._backward)
        result = SketchedLossResult(results, self.workers)
        return result
    
class SketchedLossResult(object):
    def __init__(self, tensor, workers):
        self._tensor = tensor.detach().cpu().numpy()
        self.workers = workers

    def backward(self):
        ray.wait([worker.loss_backward.remote()
            for worker in self.workers])

    def __repr__(self):
        return self._tensor.__repr__()

    def __getattr__(self, name):
        return getattr(self._tensor, name)

class SketchedOptimizer(optim.Optimizer):
    def __init__(self, optimizer, workers):
        """
        Takes in an already-initialized optimizer and list of workers (object IDs).
        Gives the workers the optimizers and then wraps optimizer methods. 
        """
        self.workers = workers
        self.head_worker = self.workers[0]
        # self.param_groups = optimizer.param_groups
        ray.wait([worker.set_optimizer.remote(optimizer) for worker in self.workers])
    
    def zero_grad(self):
        [worker.optimizer_zero_grad.remote() for worker in self.workers]

    def step(self):
        grads = [worker.compute_grad.remote() for worker in self.workers]
        ray.wait([worker.all_reduce_sketched.remote(*grads) for worker in self.workers]) 

    def __getattr__(self, name):
        if name=="param_groups":
            param_groups = ray.get(self.head_worker.get_param_groups.remote())
            print(f"Param groups are {param_groups}")
            return [SketchedParamGroup(param_group, self.workers, idx) for idx, param_group in enumerate(param_groups)]
            # param_groups = [worker.optimizer.param_groups for worker in self.workers]
            # return [SketchedParamGroup(item, self.workers) for sublist in param_groups for item in sublist]

class SketchedParamGroup(object):
    def __init__(self, param_group, workers, index):
        """
        workers is a list of object IDs
        """
        self.workers = workers
        self.param_group = param_group
        self.index = index

    def setdefault(self, name, value):
        #import pdb; pdb.set_trace()
        ray.wait([worker.param_group_setdefault.remote(self.index, name, value) for worker in self.workers])
#         [worker.remote.get_param_groups.remote()[self.index].setdefault(name, value) for worker in self.workers]
    
    def __getitem__(self, name):
        return self.param_group.__getitem__(name)
    
    def __setitem__(self, name, value):
        ray.wait([worker.param_group_setitem.remote(self.index, name, value) for worker in self.workers])
    
    #def __getattr__(self, name):
    #    return self.param_group[name]

    #def __setattr__(self, name, value):
    #    if name in ["workers", "param_group", "index"]:
    #        self.__dict__[name] = value
    #    else:
    #        ray.wait([worker.param_group_setattr.remote(self.index, name, value) for worker in self.workers])
            
    #def __getstate__(self):
    #    return self.param_group.__getstate__()

    #def __setstate__(self, state):
    #    self.param_group.__dict__.update(state)

@ray.remote(num_gpus=1)
class SketchedWorker(object):
    def __init__(self, args,
                #k=0, p2=0, num_cols=0, num_rows=0, p1=0, num_blocks=1, 
                #lr=0, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                sketch_params_larger_than=0, sketch_biases=False):
        self.num_workers = args['num_workers']
        self.k = args['k']
        self.p2 = args['p2']
        self.num_cols = args['num_cols']
        self.num_rows = args['num_rows']
        self.num_blocks = args['num_blocks']
        self.lr = args['lr']
        self.momentum = args['momentum']
        self.dampening = args['dampening']
        self.weight_decay = args['weight_decay']
        self.nesterov = args['nesterov']
        self.sketch_params_larger_than = sketch_params_larger_than
        self.sketch_biases = sketch_biases
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_model(self, model_cls, model_config, 
            sketch_biases, sketch_params_larger_than):
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        torch.random.set_rng_state(rand_state)
        model = model_cls(**model_config)
        for p in model.parameters():
            p.do_sketching = p.numel() >= sketch_params_larger_than
        # override bias terms with whatever sketchBiases is
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                if m.bias is not None:
                    m.bias.do_sketching = sketch_biases
        self.model = model.to(self.device)

    def set_loss(self, criterion):
        self.criterion = criterion.to(self.device)

    def model_call(self, *args):
        #import pdb; pdb.set_trace()
        args = [arg.to(self.device) for arg in args]
        self.outs = self.model(*args)
        return self.outs

    def model_getattr(self, name):
        return getattr(self.model, name)

    def model_setattr(self, name, value):
        if name == "model":
            self.__dict__[name] = value
        else:
            self.model.setattr(name, value)

    def param_group_setitem(self, index, name, value):
        self.param_groups[index].__setitem__(name, value)
        
    def param_group_setattr(self, index, name, value):
        self.param_groups[index].setattr(name, value)
        
    def param_group_setdefault(self, index, name, value):
        self.param_groups[index].setdefault(name, value)
        
    def get_param_groups(self):
        try:
            return [{'initial_lr': group['initial_lr'], 'lr': group['lr']} for group in self.param_groups]
        except Exception as e:
            print(f"Exception is {e}")
            return [{'lr': group['lr']} for group in self.param_groups]
    
    def loss_call(self, *args):
        args = [arg.to(self.device) for arg in args]
        #list_outs = ray.get(args[0])
        #outs = torch.stack(list_outs, dim=0)
        #import pdb; pdb.set_trace()
        #self.loss = self.criterion(args[0], args[1])
        self.loss = self.criterion(self.outs, args[1])
        #import pdb; pdb.set_trace()
        return self.loss

    def loss_backward(self):
        #import pdb; pdb.set_trace()
        self.loss.sum().backward()

    def set_optimizer(self, opt):
        assert self.model is not None, "model must be already initialized"
        p = opt.param_groups[0]
        lr = p['lr']
        dampening = p['dampening']
        nesterov = p['nesterov']
        weight_decay = p['weight_decay']
        momentum = p['momentum']
        opt = optim.SGD(self.model.parameters(), 
            lr=lr, 
            dampening=dampening, 
            nesterov=nesterov, 
            weight_decay=weight_decay, 
            momentum=momentum)
        self.param_groups = opt.param_groups
        grad_size = 0
        sketch_mask = []
        #for p in self.model.parameters():
        for group in self.param_groups:
            for p in group["params"]:
            #if True:
                if p.requires_grad:
                    size = torch.numel(p)
                    #import pdb; pdb.set_trace()
                    if p.do_sketching:
                        sketch_mask.append(torch.ones(size))
                    else:
                        sketch_mask.append(torch.zeros(size))
                    grad_size += size
        self.grad_size = grad_size
        print(f"Total dimension is {self.grad_size} using k {self.k} and p2 {self.p2}")
        self.sketch_mask = torch.cat(sketch_mask).byte().to(self.device)
        #print(f"sketchMask.sum(): {self.sketchMask.sum()}")
#         print(f"Make CSVec of dim{numRows}, {numCols}")
        self.sketch = CSVec(d=self.sketch_mask.sum().item(), 
            c=self.num_cols, 
            r=self.num_rows, 
            device=self.device, 
            nChunks=1, 
            numBlocks=self.num_blocks)
        self.u = torch.zeros(self.grad_size, device=self.device)
        self.v = torch.zeros(self.grad_size, device=self.device)

    def optimizer_zero_grad(self):
        self.zero_grad()

    def compute_grad(self):
        # compute grad 
        gradVec = self._getGradVec()
        # weight decay
        if self.weight_decay != 0:
            gradVec.add_(self.weight_decay/self.num_workers, self._getParamVec())
        if self.nesterov:
            self.u.add_(gradVec).mul_(self.momentum)
            self.v.add_(self.u).add_(gradVec)
        else:
            self.u.mul_(self.momentum).add_(gradVec)
            self.v += (self.u)
            #self.v = gradVec
        # this is v
        return self.v
        candidateSketch = self.v[self.sketchMask]
        self.sketch.zero()
        self.sketch += candidateSketch
        del candidateSketch
        # COMMUNICATE ONLY THE TABLE
        return self.sketch.table

    def all_reduce_sketched(self, *grads):
        # compute update
        grads = [grad.to(self.device) for grad in grads]
        self.sketch.zero()
        for grad in grads:
            self.sketch += grad[self.sketch_mask]
            #/len(diff_vecs)
        candidate_top_k = self.sketch.unSketch(k=self.p2*self.k)
        candidate_hh_coords = candidate_top_k.nonzero()
        hhs = [grad[candidate_hh_coords] for grad in grads]
        candidate_top_k[candidate_hh_coords] = torch.sum(
            torch.stack(hhs),dim=0)
        weights = self.topk(candidate_top_k, k=self.k)
        weight_update = torch.zeros(self.grad_size, device=self.device)
        weight_update[self.sketch_mask] = weights
        weight_update[~self.sketch_mask] = torch.sum(
            torch.stack(
                [grad[~self.sketch_mask] for grad in grads]), dim=0)
        self._apply_update(weight_update)

    def topk(self, vec, k):
        """ Return the largest k elements (by magnitude) of vec"""
        ret = torch.zeros_like(vec)

        # on a gpu, sorting is faster than pytorch's topk method
        topkIndices = torch.sort(vec**2)[1][-k:]
                        #_, topkIndices = torch.topk(vec**2, k)

        ret[topkIndices] = vec[topkIndices]
        return ret

    def _apply_update(self, update):
        # set update
        self.u[update.nonzero()] = 0
        self.v[update.nonzero()] = 0
        self.v[~self.sketch_mask] = 0
        #self.sync(weightUpdate * self._getLRVec())
        weight_update = update * self._getLRVec()
        #import pdb; pdb.set_trace()
        self._setGradVec(weight_update)
        self._updateParamsWithGradVec()

    def _getLRVec(self):
        """Return a vector of each gradient element's learning rate
        If all parameters have the same learning rate, this just
        returns torch.ones(D) * learning_rate. In this case, this
        function is memory-optimized by returning just a single
        number.
        """
        if len(self.param_groups) == 1:
            lr = self.param_groups[0]["lr"]
#            print(f"Lr is {lr}")
            return lr

        lrVec = []
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    lrVec.append(torch.zeros_like(p.data.view(-1)))
                else:
                    grad = p.grad.data.view(-1)
                    lrVec.append(torch.ones_like(grad) * lr)
        return torch.cat(lrVec)
    
    def _getGradShapes(self):
        """Return the shapes and sizes of the weight matrices"""
        with torch.no_grad():
            gradShapes = []
            gradSizes = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        gradShapes.append(p.data.shape)
                        gradSizes.append(torch.numel(p))
                    else:
                        gradShapes.append(p.grad.data.shape)
                        gradSizes.append(torch.numel(p))
            return gradShapes, gradSizes

    def _getGradVec(self):
        """Return the gradient flattened to a vector"""
        # TODO: List comprehension
        gradVec = []
        with torch.no_grad():
            # flatten
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        gradVec.append(torch.zeros_like(p.data.view(-1)))
                    else:
                        gradVec.append(p.grad.data.view(-1).float())

            # concat into a single vector
            gradVec = torch.cat(gradVec).to(self.device)

        return gradVec
    
    def _getParamVec(self):
        """Returns the current model weights as a vector"""
        d = []
        for group in self.param_groups:
            for p in group["params"]:
                d.append(p.data.view(-1).float())
        return torch.cat(d).to(self.device)
    def zero_grad(self):
        """Zero out param grads"""
        """Update params w gradient"""
        gradShapes, gradSizes = self._getGradShapes()
        startPos = 0
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                shape = gradShapes[i]
                size = gradSizes[i]
                i += 1
                if p.grad is None:
                    continue

                assert(size == torch.numel(p))
                p.grad.data.zero_()
                startPos += size
    def _setGradVec(self, vec):
        """Update params w gradient"""
        vec = vec.to(self.device)
        gradShapes, gradSizes = self._getGradShapes()
        startPos = 0
        i = 0
#         print(vec.mean())
        for group in self.param_groups:
            for p in group["params"]:
                shape = gradShapes[i]
                size = gradSizes[i]
                i += 1
                if p.grad is None:
                    continue

                assert(size == torch.numel(p))
                p.grad.data.zero_()
                p.grad.data.add_(vec[startPos:startPos + size].reshape(shape))
                startPos += size
    def sync(self, vec):
        """Set params"""
        gradShapes, gradSizes = self._getGradShapes()
        startPos = 0
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                shape = gradShapes[i]
                size = gradSizes[i]
                i += 1
                assert(size == torch.numel(p))
                p.data = vec[startPos:startPos + size].reshape(shape)
                startPos += size
    def _updateParamsWithGradVec(self):
        """Update parameters with the gradient"""
        #import pdb; pdb.set_trace()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
#                 try:
                p.data.add_(-p.grad.data)
#                 except:
#                     from IPython.core.debugger import set_trace; set_trace()

DATA_PATH = 'sample_data'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

def train(model, opt, scheduler, criterion, accuracy, 
    train_loader, val_loader, epochs, loggers=(), timer=None):
    timer = timer or Timer()
    scheduler.step()
    for epoch in range(args.epochs):
        scheduler.step()
        train_losses, train_accs = run_batches(model, opt, 
            criterion, accuracy, train_loader, True)
        train_time = timer()
        val_losses, val_accs = run_batches(model, None, 
            criterion, accuracy, val_loader, False)
        val_time = timer()
        epoch_stats = {
            'train_time': train_time,
            'train_loss': np.mean(train_losses),
            'train_acc': np.mean(train_accs),
            'test_time': val_time,
            'test_loss': np.mean(val_losses),
            'test_acc': np.mean(val_accs),
            'total_time': timer.total_time,
        }
        summary = union({'epoch': epoch+1, 'lr': scheduler.get_lr()[0]}, epoch_stats)
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
        batch_acc = accuracy(*ray.get(outs), targets).float().mean().cpu().numpy()
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
parser.add_argument("--test", type=bool, default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    "num_cols": args.cols,
    "num_rows": args.rows,
    "num_blocks": args.num_blocks,
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
ray.init(num_gpus=7, redis_password="sketched_sgd")
model_cls = Net
model_config = {}
if args.test:
    model_config = {
        'channels': {'prep': 1, 'layer1': 1, 'layer2': 1, 'layer3': 1},
    }
workers = [SketchedWorker.remote(sketched_params) for _ in range(args.num_workers)]
sketched_model = SketchedModel(model_cls, model_config, workers)
opt = optim.SGD(sketched_model.parameters(), lr=1)
sketched_opt = SketchedOptimizer(opt, workers)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
sketched_criterion = SketchedLoss(criterion, workers)
accuracy = Correct().to(device)
lambda_step = lambda t: np.interp([t], [0, 5, args.epochs+1], [0, 0.4, 0])[0]
scheduler = optim.lr_scheduler.LambdaLR(sketched_opt, lr_lambda=[lambda_step])
train(sketched_model, sketched_opt, scheduler, sketched_criterion, accuracy,
    train_loader, val_loader, args.epochs, loggers=(TableLogger(), TSVLogger()), timer=timer)
