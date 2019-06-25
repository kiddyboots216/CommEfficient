from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from core import *
from sketched_model import SketchedModel
from csvec import CSVec

import os

from worker import Worker
from core import warmup_cudnn


@ray.remote(
    num_gpus=2.0, 
    num_cpus=2.0
)
class ParameterServer(object):
    def __init__(self, kwargs):
        print(f"Received args {kwargs}")
        self.step_number = 0
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

    def compute_hhcoords(self, sketches):
        self.sketch.zero()
        self.sketch += torch.sum(torch.stack(sketches, dim=0), dim=0)
        self.candidateTopK = self.sketch.unSketch(k=self.p2*self.k)
        self.candidateHHCoords = self.candidateTopK.nonzero()
        # COMMUNICATE
        return self.candidateHHCoords

    def average_grads(self, grads):
        return torch.mean(torch.stack(grads), dim=0)

    def compute_update(self, sketchesAndUnsketched):
        sketches, unsketched = sketchesAndUnsketched
        self.candidateTopK[self.candidateHHCoords] = torch.sum(
            torch.stack(sketches),dim=0)
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

if __name__ == "__main__":
    # this unit test runs a single round of sketched SGD
    ray.init(ignore_reinit_error=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # CONSTANTS
    epochs = 1
    batch_size = 1
    D_in, D_out, H_sizes = 2, 4, [2,4]

    x = torch.randn(batch_size, D_in, device=device)
    y = torch.randn(batch_size, D_out, device=device)
    num_workers = 2
    import torch.nn as nn

    class FCNet(nn.Module):
        def __init__(self, in_size, out_size, hidden_sizes):
            super(FCNet, self).__init__()
            self.layers = nn.ModuleList()
            last_size = in_size
            for size in hidden_sizes:
                self.layers.append(nn.Linear(last_size, size))
                last_size = size
            self.final = nn.Linear(last_size, out_size)
        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))
            return self.final(x)

    model_config = {
        "in_size": D_in,
        "out_size": D_out,
        "hidden_sizes": H_sizes,
    }
    model_maker = lambda model_config: FCNet(**model_config).to(device)    
    ps = ParameterServer.remote(model_maker, model_config, num_workers, k=1, lr=1e-3, numCols=1, numRows=1, p2=1)
    # Create workers.
    workers = [Worker.remote(worker_index, model_maker, model_config, k=1, lr=1e-3, numCols=1, numRows=1, p2=1) for worker_index in range(num_workers)]
    # for _ in range(epochs):
        # workers do backward passes and calculate sketches
    sketches = [ray.get(worker.forward.remote(x, y)) for worker in workers]
        # server initiates second round of communication
    hhcoords = ray.get(ps.compute_hhcoords.remote(sketches))
        # workers answer, also giving the unsketched params
    topkAndUnsketched = list(zip(*ray.get([worker.send_topkAndUnsketched.remote(hhcoords) for worker in workers])))
        # server compute weight update, put it into ray
    weightUpdate = ray.get(ps.compute_update.remote(topkAndUnsketched))
        # workers apply weight update (can be merged with 1st line)
    ray.get([worker.apply_update.remote(weightUpdate) for worker in workers])
        # server computes validation accuracy (only one per epoch)
