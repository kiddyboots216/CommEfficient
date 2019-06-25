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
from single_trainer import SGD_Sketched
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

criterion = nn.CrossEntropyLoss(reduction='none')
correctCriterion = Correct()

def run_batches(model, batches, training, optimizer):
    stats = StatsLogger(('loss', 'correct'))
    model.train(training)
    for batchId, batch in enumerate(batches):
        inputs = batch["input"]
        targets = batch["target"]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        nCorrect = correctCriterion(outputs, targets)
        iterationStats = {"loss": loss, "correct": nCorrect}
        if training:
#             else:
            loss.sum().backward()
#                 optimizer.backward(loss)
            optimizer.step()
#             model.zero_grad()
        stats.append(iterationStats)
    return stats

def train_epoch(model, train_batches, test_batches, optimizer,
                timer, test_time_in_total=True):
    train_stats = run_batches(model, train_batches, True, optimizer)
    train_time = timer()
    test_stats = run_batches(model, test_batches, False, optimizer)
    test_time = timer(test_time_in_total)
    stats ={'train_time': train_time,
            'train_loss': train_stats.mean('loss'),
            'train_acc': train_stats.mean('correct'),
            'test_time': test_time,
            'test_loss': test_stats.mean('loss'),
            'test_acc': test_stats.mean('correct'),
            'total_time': timer.total_time}
    return stats

def train(model, optimizer, train_batches, test_batches, epochs,
          loggers=(), test_time_in_total=True, timer=None):
    timer = timer or Timer()
    for epoch in range(epochs):
        epoch_stats = train_epoch(model, train_batches, test_batches,
                                  optimizer, timer,
                                  test_time_in_total=test_time_in_total)
        lr = optimizer.param_values()['lr'] * train_batches.batch_size
        summary = union({'epoch': epoch+1, 'lr': lr}, epoch_stats)
        track.metric(iteration=epoch, **summary)
        for logger in loggers:
            logger.append(summary)
    return summary

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
        return ({'input': x.to(device), 'target': y.to(device).long()}
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

trainable_params = lambda model: filter(lambda p: p.requires_grad, model.parameters())

class TorchOptimiser():
    def __init__(self, weights, optimizer, sketched,
                 k, p2, numCols, numRows, step_number=0, **opt_params):
        self.weights = weights
        self.step_number = step_number
        self.opt_params = opt_params
        self._opt = optimizer(weights, **self.param_values())
        if sketched:
            assert(optimizer == torch.optim.SGD)
            assert(opt_params["dampening"] == 0)
#             assert(opt_params["nesterov"] == False)
            self._opt = SGD_Sketched(weights, k, p2, numCols, numRows, 
                                     **self.param_values())

    def param_values(self):
        return {k: v(self.step_number) if callable(v) else v
                for k,v in self.opt_params.items()}

    def step(self, loss=None):
        self.step_number += 1
        self._opt.param_groups[0].update(**self.param_values())
        self._opt.step(loss)

    def __repr__(self):
        return repr(self._opt)
    
    def __getattr__(self, key):
        return getattr(self._opt, key)

def SGD(weights, lr, momentum, weight_decay, nesterov, dampening,
        sketched, k, p2, numCols, numRows, numBlocks):
    return TorchOptimiser(weights, torch.optim.SGD, sketched=sketched, k=k, p2=p2, 
                          numCols=numCols, numRows=numRows, lr=lr,
                          momentum=momentum, weight_decay=weight_decay,
                          dampening=dampening, nesterov=nesterov)

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
