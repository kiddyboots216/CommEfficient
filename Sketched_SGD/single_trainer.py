import math
import torch
from torch.optim import Optimizer
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from csvec import CSVec
from sketched_model import SketchedModel, topk

class SGD_Sketched(Optimizer):

    def __init__(self, params, k, p2, numCols, numRows, lr,
                 momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 numBlocks=1, p1=0):
        """
        First line of args are required.
        Second line of args are SGD args.
        Third line of args are Sketched args.
        > optim = SGD_Sketched(model.parameters(), k=5e4, lr=1e-3, numCols=5e5,
                               numRows=5, p2=4, momentum=0.9, numBlocks=4)
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       nesterov=nesterov)
        super(SGD_Sketched, self).__init__(params, defaults)
        # set device
        if self.param_groups[0]["params"][0].is_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
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
        self.sketchMask = torch.cat(sketchMask).byte().to(self.device)
#         self.us = [torch.zeros(self.grad_size, device=self.device) for _ in range(1)]
#         self.vs = [torch.zeros(self.grad_size, device=self.device) for _ in range(1)]
        self.u = torch.zeros(self.grad_size, device=self.device)
        self.v = torch.zeros(self.grad_size, device=self.device)
#         self.sketches = [CSVec(d=self.sketchMask.sum().item(), c=numCols, r=numRows, 
#                             device=self.device, nChunks=1, numBlocks=numBlocks) for _ in range(1)]
        self.sketch = CSVec(d=self.sketchMask.sum().item(), c=numCols, r=numRows, 
                            device=self.device, nChunks=1, numBlocks=numBlocks)

    def __setstate__(self, state):
        super(SGD_Sketched, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
#     def zero_grad(self):
#
    def step(self, loss=None):
        """Performs a single optimization step.

        Arguments:
            loss: to backprop
        backward
            _backwardWorker
            _aggregateAndZeroUVs
                _aggAndZeroSketched
        """
#         initialGradVec = self._getGradVec()
#         self._setGradVec(torch.zeros_like(initialGradVec))
        """
        Calls _backwardWorker inside backward
        """
        # flatten all grads for now
#         self.zero_grad()
#         loss.sum().backward()
        gradVec = self._getGradVec()
        # weight decay
        if self.weight_decay != 0:
            gradVec.add_(self.weight_decay, self._getParamVec())
        # TODO: Pretty sure this momentum/residual formula is wrong
        if self.nesterov:
            self.u.add_(gradVec).mul_(self.momentum)
            self.v.add_(self.u).add_(gradVec)
#             self.us[0].add_(gradVec).mul_(self.momentum)
#             self.vs[0].add_(self.us[0]).add_(gradVec)
        else:
            self.u.mul_(self.momentum).add_(gradVec)
            self.v.add_(self.u)
#             self.us[0].mul_(self.momentum).add_(gradVec)
#             self.vs[0].add_(self.us[0])
        """
        Calls _aggAndZeroSketched inside _aggregateAndZeroUVs
        """
        # this is v
#         vs = [v[self.sketchMask] for v in self.vs]
        candidateSketch = self.v[self.sketchMask]
        self.sketch.zero()
        self.sketch += candidateSketch
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
        
    """
    Helper functions below
    """
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
                p.data.add_(-p.grad.data)

if __name__ == "__main__":
    # this unit tests does one optimization pass of the single/centralized trainer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    D_in, D_out, H_sizes = 4, 1, [100, 100]
    model = FCNet(D_in, D_out, H_sizes).to(device)
    sketchedModel = SketchedModel(model)
    optim = SGD_Sketched(sketchedModel.parameters(), k=10, lr=1e-3, numCols=100, numRows=5, p2=4) 
    # CONSTANTS
    epochs = 1
    batch_size = 32
    x = torch.randn(batch_size, D_in, device=device)
    y = torch.randn(batch_size, D_out, device=device)
    optim.zero_grad()
    criterion = torch.nn.MSELoss(reduction='sum')
    loss = criterion(model(x), y)
    loss.backward()
    optim.step()







