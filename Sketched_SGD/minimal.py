
#                 except:
#                     from IPython.core.debugger import set_trace; set_trace()


import ray
import time
import torch
import torch.nn as nn
#from core import Net
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1, 1)
"""
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
optim_args = {
        "k": args.k,
        "p2": args.p2,
        "p1": args.p1,
        "numCols": args.cols,
        "numRows": args.rows,
        "numBlocks": args.num_blocks,
        "lr": lambda step: 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4*args.batch_size,
        "nesterov": args.nesterov,
        "dampening": 0,
        }
ray.init(num_gpus=8)
num_workers = 4
#workers = [Worker.remote(num_workers, worker_index, optim_args) for worker_index in range(num_workers)]
#ray.wait([worker.step.remote() for worker in workers])
counters = [Counter.remote() for _ in range(4)]
ray.wait([counter.train.remote() for counter in counters])
