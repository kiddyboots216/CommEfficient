import torch
import torch.nn as nn

class SketchedModel:
    # not inheriting from nn.Module to avoid the fact that implementing
    # __getattr__ on a nn.Module is tricky, since self.model = model
    # doesn't actually add "model" to self.__dict__ -- instead, nn.Module
    # creates a key/value pair in some internal dictionary that keeps
    # track of submodules
    def __init__(self, model, sketchBiases=False, sketchParamsLargerThan=0):
        self.model = model
        # sketch everything larger than sketchParamsLargerThan
        for p in model.parameters():
            p.do_sketching = p.numel() >= sketchParamsLargerThan

        # override bias terms with whatever sketchBiases is
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                if m.bias is not None:
                    m.bias.do_sketching = sketchBiases

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        if name == "model":
            self.__dict__[name] = value
        else:
            self.model.setattr(name, value)
            
def topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    ret = torch.zeros_like(vec)

    # on a gpu, sorting is faster than pytorch's topk method
    topkIndices = torch.sort(vec**2)[1][-k:]
    #_, topkIndices = torch.topk(vec**2, k)

    ret[topkIndices] = vec[topkIndices]
    return ret


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
        print(*('{:12.4f}'.format(v)
                 if isinstance(v, np.float) else '{:12}'.format(v)
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


#####################
## training utils
#####################

@singledispatch
def cat(*xs):
    raise NotImplementedError

@singledispatch
def to_numpy(x):
    raise NotImplementedError


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

class StatsLogger():
    def __init__(self, keys):
        self._stats = {k:[] for k in keys}

    def append(self, output):
        for k,v in self._stats.items():
            v.append(output[k].detach())

    def stats(self, key):
        return cat(*self._stats[key])

    def mean(self, key):
        return np.mean(to_numpy(self.stats(key)), dtype=np.float)

# criterion = nn.CrossEntropyLoss(reduction='none')
# correctCriterion = Correct()

# def run_batches(model, batches, training, optimizer):
#     stats = StatsLogger(('loss', 'correct'))
#     model.train(training)
#     for batchId, batch in enumerate(batches):
#         inputs = batch["input"]
#         targets = batch["target"]
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         nCorrect = correctCriterion(outputs, targets)
#         iterationStats = {"loss": loss, "correct": nCorrect}
#         if training:
# #             else:
#             loss.sum().backward()
# #                 optimizer.backward(loss)
#             optimizer.step()
# #             model.zero_grad()
#         stats.append(iterationStats)
#     return stats

# def train_epoch(model, train_batches, test_batches, optimizer,
#                 timer, test_time_in_total=True):
#     train_stats = run_batches(model, train_batches, True, optimizer)
#     train_time = timer()
#     test_stats = run_batches(model, test_batches, False, optimizer)
#     test_time = timer(test_time_in_total)
#     stats ={'train_time': train_time,
#             'train_loss': train_stats.mean('loss'),
#             'train_acc': train_stats.mean('correct'),
#             'test_time': test_time,
#             'test_loss': test_stats.mean('loss'),
#             'test_acc': test_stats.mean('correct'),
#             'total_time': timer.total_time}
#     return stats

# def train(model, optimizer, train_batches, test_batches, epochs,
#           loggers=(), test_time_in_total=True, timer=None):
#     timer = timer or Timer()
#     for epoch in range(epochs):
#         epoch_stats = train_epoch(model, train_batches, test_batches,
#                                   optimizer, timer,
#                                   test_time_in_total=test_time_in_total)
#         lr = optimizer.param_values()['lr'] * train_batches.batch_size
#         summary = union({'epoch': epoch+1, 'lr': lr}, epoch_stats)
#         track.metric(iteration=epoch, **summary)
#         for logger in loggers:
#             logger.append(summary)
    # return summary

#####################
## network visualisation (requires pydot)
#####################
class ColorMap(dict):
    palette = (
        'bebada,ffffb3,fb8072,8dd3c7,80b1d3,fdb462,b3de69,fccde5,bc80bd,ccebc5,ffed6f,1f78b4,33a02c,e31a1c,ff7f00,'
        '4dddf8,e66493,b07b87,4e90e3,dea05e,d0c281,f0e189,e9e8b1,e0eb71,bbd2a4,6ed641,57eb9c,3ca4d4,92d5e7,b15928'
    ).split(',')
    def __missing__(self, key):
        self[key] = self.palette[len(self) % len(self.palette)]
        return self[key]

def make_pydot(nodes, edges, direction='LR', sep=sep, **kwargs):
    import pydot
    parent = lambda path: path[:-1]
    stub = lambda path: path[-1]
    class Subgraphs(dict):
        def __missing__(self, path):
            subgraph = pydot.Cluster(sep.join(path), label=stub(path),
                                     style='rounded, filled',
                                     fillcolor='#77777744')
            self[parent(path)].add_subgraph(subgraph)
            return subgraph
    subgraphs = Subgraphs()
    subgraphs[()] = g = pydot.Dot(rankdir=direction, directed=True,
                                  **kwargs)
    g.set_node_defaults(
        shape='box', style='rounded, filled', fillcolor='#ffffff')
    for node, attr in nodes:
        path = tuple(node.split(sep))
        subgraphs[parent(path)].add_node(
            pydot.Node(name=node, label=stub(path), **attr))
    for src, dst, attr in edges:
        g.add_edge(pydot.Edge(src, dst, **attr))
    return g

get_params = lambda mod: {p.name: getattr(mod, p.name, '?')
                          for p in signature(type(mod)).parameters.values()
                         }


class DotGraph():
    colors = ColorMap()
    def __init__(self, net, size=15, direction='LR'):
        graph = build_graph(net)
        self.nodes = [(k, {
            'tooltip': '%s %.1000r' % (type(n).__name__, get_params(n)),
            'fillcolor': '#'+self.colors[type(n)],
        }) for k, (n, i) in graph.items()]
        self.edges = [(src, k, {})
                      for (k, (n, i)) in graph.items()
                      for src in i]
        self.size, self.direction = size, direction

    def dot_graph(self, **kwargs):
        return make_pydot(self.nodes, self.edges, size=self.size,
                            direction=self.direction, **kwargs)

    def svg(self, **kwargs):
        return self.dot_graph(**kwargs).create(format='svg').decode('utf-8')

    try:
        import pydot
        def _repr_svg_(self):
            return self.svg()
    except ImportError:
        def __repr__(self):
            return 'pydot is needed for network visualisation'

walk = lambda dict_, key: walk(dict_, dict_[key]) if key in dict_ else key

def remove_by_type(net, node_type):
    #remove identity nodes for more compact visualisations
    graph = build_graph(net)
    remap = {k: i[0]
             for k,(v,i) in graph.items()
             if isinstance(v, node_type)}
    return {k: (v, [walk(remap, x) for x in i])
            for k, (v,i) in graph.items()
            if not isinstance(v, node_type)}

import numpy as np
import torch
from torch import nn
import torchvision
# from core import cat, to_numpy
#from core import build_graph

# torch.backends.cudnn.benchmark = True
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
@cat.register(torch.Tensor)
def _(*xs):
    return torch.cat(xs)

@to_numpy.register(torch.Tensor)
def _(x):
    return x.detach().cpu().numpy()

def warmup_cudnn(model, batch_size):
    #run forward and backward pass of the model on a batch of random inputs
    #to allow benchmarking of cudnn kernels
    inp = torch.Tensor(np.random.rand(batch_size, 3, 32, 32)).cuda()
    target = torch.LongTensor(np.random.randint(0, 10, batch_size)).cuda()
#     batch = {'input': inp, 'target': target}
#     model.train(True)
    o = model(inp)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    loss = criterion(o, target)
    loss.backward()
    model.zero_grad()
    torch.cuda.synchronize()

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



"""
class Network(nn.Module):
    def __init__(self, net):
        self.graph = build_graph(net)
        super().__init__()
        for n, (v, _) in self.graph.items():
            setattr(self, n, v)
    def forward(self, inputs):
        self.cache = dict(inputs)
        for n, (_, i) in self.graph.items():
            #import ipdb
            #ipdb.set_trace()
            print(n)
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache
    def half(self):
        for module in self.children():
            if not isinstance(module, nn.BatchNorm2d):
                module.half()
        return self
"""

# trainable_params = lambda model: filter(lambda p: p.requires_grad, model.parameters())

# class TorchOptimiser():
#     def __init__(self, weights, optimizer, sketched,
#                  k, p2, numCols, numRows, step_number=0, **opt_params):
#         self.weights = weights
#         self.step_number = step_number
#         self.opt_params = opt_params
#         self._opt = optimizer(weights, **self.param_values())
#         if sketched:
#             assert(optimizer == torch.optim.SGD)
#             assert(opt_params["dampening"] == 0)
# #             assert(opt_params["nesterov"] == False)
#             self._opt = SGD_Sketched(weights, k, p2, numCols, numRows, 
#                                      **self.param_values())

#     def param_values(self):
#         return {k: v(self.step_number) if callable(v) else v
#                 for k,v in self.opt_params.items()}

#     def step(self, loss=None):
#         self.step_number += 1
#         self._opt.param_groups[0].update(**self.param_values())
#         self._opt.step(loss)

#     def __repr__(self):
#         return repr(self._opt)
    
#     def __getattr__(self, key):
#         return getattr(self._opt, key)

# def SGD(weights, lr, momentum, weight_decay, nesterov, dampening,
#         sketched, k, p2, numCols, numRows, numBlocks):
#     return TorchOptimiser(weights, torch.optim.SGD, sketched=sketched, k=k, p2=p2, 
#                           numCols=numCols, numRows=numRows, lr=lr,
#                           momentum=momentum, weight_decay=weight_decay,
#                           dampening=dampening, nesterov=nesterov)

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

import argparse
import numpy as np

import ray
import torch
from collections import defaultdict
from copy import deepcopy
from itertools import chain

import math
import torch
from torch.optim import Optimizer
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# from core import *
# from sketched_model import SketchedModel
# from csvec import CSVec

# import os

# from worker import Worker
# from core import warmup_cudnn


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()
@ray.remote(num_gpus=2.0)
class ParameterServer(object):
    def __init__(self, kwargs):
        print(f"Received args {kwargs}")
        self.step_number = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in ray.get_gpu_ids()])
        print(os.environ["CUDA_VISIBLE_DEVICES"])
        self.params = kwargs
        self.sketcher_init(**self.param_values())
        # super().__init__(**self.param_values())
        warmed_up = False
        while not warmed_up:
            try:
                for size in [512, 256]:
                        warmup_cudnn(self.sketchedModel, size)
                warmed_up = True
            except RuntimeError as e:
                print(e)
        del self.sketchedModel
        del self.param_groups
    
    def sketcher_init(self, 
                 k=0, p2=0, numCols=0, numRows=0, p1=0, numBlocks=1, # sketched_params
                 lr=0, momentum=0, dampening=0, weight_decay=0, nesterov=False): # opt_params
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids())) 
        print(ray.get_gpu_ids())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net().cuda()
        self.sketchedModel = SketchedModel(model)
        trainable_params = lambda model: filter(lambda p: p.requires_grad, model.parameters())
        params = trainable_params(self.sketchedModel)
#         params = sketchedModel.parameters()
        # checking before default Optimizer init
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       nesterov=nesterov)
        # default Optimizer initialization
        self.defaults = defaults

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)
        # SketchedSGD-specific
        # set device
        #self.device = model_config
#         print(f"I am using backend of {self.device}")
#         if self.param_groups[0]["params"][0].is_cuda:
#             self.device = "cuda:0"
#         else:
#             self.device = "cpu"
        # set all the regular SGD params as instance vars
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        # set the sketched params as instance vars
        self.p2 = p2
        self.k = k
        # initialize sketchMask, sketch, momentum buffer and accumulated grads
        # this is D
        grad_size = 0
        sketchMask = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    size = torch.numel(p)
                    if p.do_sketching:
                        sketchMask.append(torch.ones(size))
                    else:
                        sketchMask.append(torch.zeros(size))
                    grad_size += size
        self.grad_size = grad_size
        print(f"Total dimension is {self.grad_size}")
        self.sketchMask = torch.cat(sketchMask).byte().to(self.device)
        print(f"sketchMask.sum(): {self.sketchMask.sum()}")
#         print(f"Make CSVec of dim{numRows}, {numCols}")
        self.sketch = CSVec(d=self.sketchMask.sum().item(), c=numCols, r=numRows, 
                            device=self.device, nChunks=1, numBlocks=numBlocks)

    def compute_hhcoords(self, tables):
        self.sketch.zero()
        self.sketch.table = torch.sum(torch.stack(tables, dim=0), dim=0)
        self.candidateTopK = self.sketch.unSketch(k=self.p2*self.k)
        self.candidateHHCoords = self.candidateTopK.nonzero()
        # COMMUNICATE
        return self.candidateHHCoords

    def average_grads(self, grads):
        return torch.sum(torch.stack(grads), dim=0)

    def compute_update(self, sketchesAndUnsketched):
        hhs, unsketched = sketchesAndUnsketched
        self.candidateTopK[self.candidateHHCoords] = torch.sum(
            torch.stack(hhs),dim=0)
        del self.candidateHHCoords
        weights = self.topk(self.candidateTopK, k=self.k)
        del self.candidateTopK
        weightUpdate = torch.zeros(self.grad_size, device=self.device)
        weightUpdate[self.sketchMask] = weights
        weightUpdate[~self.sketchMask] = torch.sum(torch.stack(unsketched), dim=0)
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

    """
    Helper functions below
    """
    def param_values(self):
#         print(f"Kwargs are {self.params}")
        params = {k: v(self.step_number) if callable(v) else v
                for k,v in self.params.items()}
#         print(f"Params are {params}")
        return params
    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
        
    def _getLRVec(self):
        """Return a vector of each gradient element's learning rate
        If all parameters have the same learning rate, this just
        returns torch.ones(D) * learning_rate. In this case, this
        function is memory-optimized by returning just a single
        number.
        """
        if len(self.param_groups) == 1:
            return self.param_groups[0]["lr"]

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
    def _updateParamsWithGradVec(self):
        """Update parameters with the gradient"""
        for group in self.param_groups:
            for p in group["params"]:
#                if p.grad is None:
#                    continue
#                 try:
                p.data.add_(-p.grad.data)
#                 except:
#                     from IPython.core.debugger import set_trace; set_trace()

import numpy as np

import ray
import torch
from collections import defaultdict
from copy import deepcopy
from itertools import chain

import math
import torch
from torch.optim import Optimizer
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# from core import *
# from sketched_model import SketchedModel
# from csvec import CSVec

import os


# from sketcher import Sketcher
# from core import warmup_cudnn


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()
class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

@ray.remote(num_gpus=1.0)
class Worker(object):
    def __init__(self, num_workers, worker_index, kwargs):
        #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in ray.get_gpu_ids()])
        #print(os.environ["CUDA_VISIBLE_DEVICES"])
        self.worker_index = worker_index 
        self.num_workers = num_workers
        self.step_number = 0
        self.params = kwargs
        print(f"Initializing worker {self.worker_index}")
        self.sketcher_init(**self.param_values())
        warmed_up = False
        while not warmed_up:
            try:
                for size in [512, 256]:
                        warmup_cudnn(self.sketchedModel, size)
                warmed_up = True
            except RuntimeError as e:
                print(e)
        self.u = torch.zeros(self.grad_size, device=self.device)
        self.v = torch.zeros(self.grad_size, device=self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        self.correctCriterion = Correct().cuda()

    def centralized_step(self):
        gradVec = self._getGradVec()
        # weight decay
        if self.weight_decay != 0:
            gradVec.add_(self.weight_decay, self._getParamVec())
        # TODO: Pretty sure this momentum/residual formula is wrong
        if self.nesterov:

            self.u.mul_(self.momentum).add_(gradVec)
            self.v.add_(self.momentum, self.u)
        else:
            self.u.mul_(self.momentum).add_(gradVec)
            self.v.add_(self.u)
        weightUpdate = self.v
        self.v = torch.zeros_like(self.v, device=self.device)
        self._setGradVec(weightUpdate * self._getLRVec())
        self._updateParamsWithGradVec()
        # return
        # candidateSketch = self.v[self.sketchMask]
        # self.sketch.zero()
        # self.sketch += candidateSketch
        # COMMUNICATE
#         for workerId, v in enumerate(vs):
#         # zero last sketch
#             self.sketches[workerId].zero()
#             # update sketch without truncating, this calls CSVec.__iadd__
#             self.sketches[workerId] += v
        # 2nd round of communication
        # don't need to sum
        
        # THIS ON SERVER
        candidateTopK = self.sketch.unSketch(k=self.p2*self.k)
#         candidateTopK = np.sum(self.sketches).unSketch(k=self.p2*self.k)
        candidateHHCoords = candidateTopK.nonzero()
        # don't need to stack or sum
        # COMMUNICATE
        candidateTopK[candidateHHCoords] = candidateSketch[candidateHHCoords]
#         candidateTopK[candidateHHCoords] = torch.sum(torch.stack([v[candidateHHCoords]
#                                                     for v in vs]),
#                                                     dim=0)
#         del vs
        del candidateSketch
        # this is w
        weights = topk(candidateTopK, k=self.k)
        del candidateTopK
        weightUpdate = torch.zeros_like(self.v)
#         weightUpdate = torch.zeros_like(self.vs[0])
        weightUpdate[self.sketchMask] = weights
        # zero out the coords that are getting communicated
        self.u[weightUpdate.nonzero()] = 0
        self.v[weightUpdate.nonzero()] = 0
#         for u, v in zip(self.us, self.vs):
#             u[weightUpdate.nonzero()] = 0
#             v[weightUpdate.nonzero()] = 0
        """
        Return from _aggAndZeroSketched, finish _aggregateAndZeroUVs
        """
        # TODO: Bundle this efficiently
        # directly send whatever wasn't sketched
        unsketched = self.v[~self.sketchMask]
#         vs = [v[~self.sketchMask] for v in self.vs]
        # don't need to sum
        
        weightUpdate[~self.sketchMask] = unsketched
#         weightUpdate[~self.sketchMask] = torch.sum(torch.stack(vs), dim=0)
#         print(torch.sum(weightUpdate))
        self.v[~self.sketchMask] = 0
#         for v in self.vs:
#             v[~self.sketchMask] = 0
        """
        Return from _aggregateAndZeroUVs, back in backward
        """
        self._setGradVec(weightUpdate * self._getLRVec())
        self._updateParamsWithGradVec()

    def sketcher_init(self, 
                 k=0, p2=0, numCols=0, numRows=0, p1=0, numBlocks=1, # sketched_params
                 lr=0, momentum=0, dampening=0, weight_decay=0, nesterov=False): # opt_params
        #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids())) 
        #print(ray.get_gpu_ids())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net().cuda()
        self.sketchedModel = SketchedModel(model)
        trainable_params = lambda model: filter(lambda p: p.requires_grad, model.parameters())
        params = trainable_params(self.sketchedModel)
#         params = sketchedModel.parameters()
        # checking before default Optimizer init
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       nesterov=nesterov)
        # default Optimizer initialization
        self.defaults = defaults

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)
        # set all the regular SGD params as instance vars
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        # set the sketched params as instance vars
        self.p2 = p2
        self.k = k
        # initialize sketchMask, sketch, momentum buffer and accumulated grads
        # this is D
        grad_size = 0
        sketchMask = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    size = torch.numel(p)
                    if p.do_sketching:
                        sketchMask.append(torch.ones(size))
                    else:
                        sketchMask.append(torch.zeros(size))
                    grad_size += size
        self.grad_size = grad_size
        print(f"Total dimension is {self.grad_size} using k {self.k} and p2 {self.p2}")
        self.sketchMask = torch.cat(sketchMask).byte().to(self.device)
        #print(f"sketchMask.sum(): {self.sketchMask.sum()}")
#         print(f"Make CSVec of dim{numRows}, {numCols}")
        self.sketch = CSVec(d=self.sketchMask.sum().item(), c=numCols, r=numRows, 
                            device=self.device, nChunks=1, numBlocks=numBlocks)


    # below two functions are only used for debugging to confirm that this works when we send full grad
    def step(self):
        self.step_number += 1
        self.param_groups[0].update(**self.param_values())
        gradVec = self._getGradVec()
        
        #return gradVec
        # weight decay
        if self.weight_decay != 0:
            gradVec.add_(self.weight_decay, self._getParamVec())
        # TODO: Pretty sure this momentum/residual formula is wrong
        self.u.mul_(self.momentum).add_(1, gradVec)
        gradVec = gradVec.add_(self.momentum, self.u)
        return gradVec
        weightUpdate = self.v
        return weightUpdate
    def update(self, weightUpdate):
        #import ipdb; ipdb.set_trace()
        self._setGradVec(weightUpdate * self._getLRVec())
        self._updateParamsWithGradVec()
        #self.v = torch.zeros_like(self.v, device=self.device)
        return

    def forward(self, inputs, targets, training=True):
        #self.sketchedModel.train(training)
        self.zero_grad()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.sketchedModel(inputs)
        loss = self.criterion(outputs, targets)
        accuracy = self.correctCriterion(outputs, targets)
        sketch = "bleh"
        if training:
            sketch = self.compute_sketch()
            loss.sum().backward()
        return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy(), sketch
    
    def compute_sketch(self): 
        """
        Calls _backwardWorker inside backward
        """
        self.step_number += 1
        self.param_groups[0].update(**self.param_values())
        gradVec = self._getGradVec()
        self.sketch.zero()
        # weight decay
        if self.weight_decay != 0:
            gradVec.add_(self.weight_decay, self._getParamVec())
        if self.nesterov:
            self.u.add_(gradVec).mul_(self.momentum)
            self.v.add_(self.u).add_(gradVec)
        else:
            self.u.mul_(self.momentum).add_(gradVec)
            self.v += (self.u)
            self.v = gradVec
        # this is v
        self.candidateSketch = self.v[self.sketchMask]
        self.sketch += self.candidateSketch
        # COMMUNICATE ONLY THE TABLE
        import pdb; pdb.set_trace()
        return self.sketch.table

    def send_topkAndUnsketched(self, hhcoords):
    #    hhcoords = hhcoords.to(self.device)
        # directly send whatever wasn't sketched
        unsketched = self.v[~self.sketchMask]
        # COMMUNICATE
        return self.v[hhcoords], unsketched
    #.cpu()
#     @ray.remote
    def apply_update(self, weightUpdate):
        # zero out the coords that are getting communicated
        self.u[weightUpdate.nonzero()] = 0
        self.v[weightUpdate.nonzero()] = 0
        self.v[~self.sketchMask] = 0
        self._setGradVec(weightUpdate * self._getLRVec())
        self._updateParamsWithGradVec()
        #self.sketchedModel.zero_grad()

    """
    Helper functions below
    """
    def param_values(self):
#         print(f"Kwargs are {self.params}")
        params = {k: v(self.step_number) if callable(v) else v
                for k,v in self.params.items()}
#         print(f"Params are {params}")
        return params
    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
        
    def _getLRVec(self):
        """Return a vector of each gradient element's learning rate
        If all parameters have the same learning rate, this just
        returns torch.ones(D) * learning_rate. In this case, this
        function is memory-optimized by returning just a single
        number.
        """
        if len(self.param_groups) == 1:
            return self.param_groups[0]["lr"]

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
    def _updateParamsWithGradVec(self):
        """Update parameters with the gradient"""
        for group in self.param_groups:
            for p in group["params"]:
#                if p.grad is None:
#                    continue
#                 try:
                p.data.add_(-p.grad.data)
#                 except:
#                     from IPython.core.debugger import set_trace; set_trace()
    
@ray.remote(num_gpus=1)
class Counter(object):
    def __init__(self):
        self.counter = torch.zeros(10).cuda()
        self.net = Net().cuda()
        print("hello world")
        time.sleep(1)
    def train(self):
        for i in range(10):
            time.sleep(1)
            self.counter += torch.ones(10).cuda()
            print(i)

from inspect import signature
from collections import namedtuple
import time
import numpy as np
import pandas as pd
from functools import singledispatch
from collections import OrderedDict
import track
import ray
import torch
import torch.nn as nn
import math

# ALL THE STUFF THAT BREAKS

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

class StatsLogger():
    def __init__(self, keys):
        self.stats = {k:[] for k in keys}

    def append(self, output):
        for k,v in self.stats.items():
            v.append(output[k])
#             v.append(output[k].detach())

#     def stats(self, key):
#         return cat(*self._stats[key])

    def mean(self, key):
        return np.mean(self.stats[key], dtype=np.float)
#         return np.mean(to_numpy(self.stats(key)), dtype=np.float)

def run_batches(ps, workers, batches, minibatch_size, training):
    stats = StatsLogger(('loss', 'correct'))
#     model.train(training)
    for batchId, batch in enumerate(batches):
        inputs = batch["input"]
        targets = batch["target"]    
        input_minibatches = []
        target_minibatches = []
        batch_size = len(inputs)
        num_workers = len(workers)
        for i, _ in enumerate(workers):
            start = i * batch_size // num_workers
            end = (i+1) * batch_size // num_workers
            input_minibatches.append(inputs[start:end])
            target_minibatches.append(targets[start:end])
        # workers do backward passes and calculate sketches
        losses, accuracies, sketches = list(zip(
            *ray.get(
                [worker.forward.remote(
                input_minibatches[worker_id],
                target_minibatches[worker_id],
                training)
                for worker_id, worker in enumerate(workers)]
                )
            ))
        if training:
            ray.wait([worker.centralized_step() for worker in workers])
            """
            weights = ray.get([worker.step.remote() for worker in workers])
            weightUpdate = ps.average_grads.remote(weights) 
            ray.wait([worker.update.remote(weightUpdate) for worker in workers])
            """
            """
            # server initiates second round of communication
            hhcoords = ps.compute_hhcoords.remote((sketches))
            # workers answer, also giving the unsketched params
            topkAndUnsketched = list(zip(
                *ray.get(
                    [worker.send_topkAndUnsketched.remote(hhcoords) for worker in workers]
                    )
                ))
            # server compute weight update, put it into ray
            weightUpdate = ps.compute_update.remote(topkAndUnsketched)
            # workers apply weight update (can be merged with 1st line)
            ray.wait([worker.apply_update.remote(weightUpdate) for worker in workers])
            """
        iterationStats = {"loss": np.mean((losses)), "correct": np.mean((accuracies))}
        #print(iterationStats)
        stats.append(iterationStats)
    return stats
#"""
def train_epoch(ps, workers, train_batches, test_batches, minibatch_size,
                timer, test_time_in_total=True):
    train_stats = run_batches(ps, workers, train_batches, minibatch_size, True)
    train_time = timer()
    test_stats = run_batches(ps, workers, test_batches, minibatch_size, False)
    test_time = timer(test_time_in_total)
    stats ={'train_time': train_time,
            'train_loss': train_stats.mean('loss'),
            'train_acc': train_stats.mean('correct'),
            'test_time': test_time,
            'test_loss': test_stats.mean('loss'),
            'test_acc': test_stats.mean('correct'),
            'total_time': timer.total_time}
    return stats

def train(ps, workers, train_batches, test_batches, epochs, minibatch_size,
          loggers=(), test_time_in_total=True, timer=None):
    timer = timer or Timer()
    for epoch in range(epochs):
        epoch_stats = train_epoch(ps, workers, train_batches, test_batches, minibatch_size,
                                  timer,
                                  test_time_in_total=test_time_in_total)
        lr = ray.get(workers[0].param_values.remote())['lr'] * train_batches.batch_size
        summary = union({'epoch': epoch+1, 'lr': lr}, epoch_stats)
#         track.metric(iteration=epoch, **summary)
        for logger in loggers:
            logger.append(summary)
    return summary

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sketched", action="store_true")
parser.add_argument("--sketch_biases", action="store_true")
parser.add_argument("--sketch_params_larger_than", action="store_true")
parser.add_argument("-k", type=int, default=50000)
parser.add_argument("--p2", type=int, default=1)
parser.add_argument("--p1", type=int, default=0)
parser.add_argument("--cols", type=int, default=500000)
parser.add_argument("--rows", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_blocks", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--epochs", type=int, default=24)
parser.add_argument("--test", action="store_true")
args = parser.parse_args()
#args.batch_size = math.ceil(args.batch_size/args.num_workers) * args.num_workers
if args.test:
    args.k = 50
    args.cols = 500
    model_maker = lambda model_config: Net(
    {'prep': 1, 'layer1': 2,
                                 'layer2': 4, 'layer3': 8}
    ).to(model_config["device"])
else:
    model_maker = lambda model_config: Net().to(model_config["device"])
model_config = {
#     "device": "cpu",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

print('Downloading datasets')
DATA_DIR = "sample_data"
dataset = cifar10(DATA_DIR)

lr_schedule = PiecewiseLinear([0, 5, args.epochs], [0, 0.4, 0])
train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
lr = lambda step: lr_schedule(step/len(train_batches))/args.batch_size
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

TSV = TSVLogger()

train_batches = Batches(Transform(train_set, train_transforms),
                        args.batch_size, shuffle=True,
                        set_random_choices=True, drop_last=True)
test_batches = Batches(test_set, args.batch_size, shuffle=False,
                       drop_last=False)

optim_args = {
    "k": args.k,
    "p2": args.p2,
    "p1": args.p1,
    "numCols": args.cols,
    "numRows": args.rows,
    "numBlocks": args.num_blocks,
    "lr": lr,
    "momentum": 0.9,
    "weight_decay": 5e-4*args.batch_size,
    "nesterov": args.nesterov,
    "dampening": 0,
}

ray.init(num_gpus=8)
num_workers = args.num_workers
minibatch_size = args.batch_size/num_workers
print(f"Passing in args {optim_args}")
ps = ParameterServer.remote(optim_args)
# Create workers.
workers = [Worker.remote(num_workers, worker_index, optim_args) for worker_index in range(num_workers)]

# track_dir = "sample_data"
# with track.trial(track_dir, None, param_map=vars(optim_args)):

train(ps, workers, train_batches, test_batches, args.epochs, minibatch_size,
      loggers=(TableLogger(), TSV), timer=timer,
      test_time_in_total=False)
