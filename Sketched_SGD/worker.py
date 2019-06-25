import torch
import torch.nn as nn
import ray

from sketcher import Sketcher
from core import warmup_cudnn


class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

@ray.remote(
    num_gpus=1.0, 
    num_cpus=2.0,
)
class Worker(Sketcher):
    def __init__(self, num_workers, worker_index, kwargs):
        self.worker_index = worker_index 
        self.num_workers = num_workers
        self.step_number = 0
        self.params = kwargs
        print(f"Initializing worker {self.worker_index}")
        super().__init__(**self.param_values())
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

    # below two functions are only used for debugging to confirm that this works when we send full grad
    def step(self):
        self.step_number += 1
        self.param_groups[0].update(**self.param_values())
        gradVec = self._getGradVec()
        # weight decay
        if self.weight_decay != 0:
            gradVec.add_(self.weight_decay, self._getParamVec())
        # TODO: Pretty sure this momentum/residual formula is wrong
        self.u.mul_(self.momentum).add_(gradVec)
        self.v.add_(self.momentum, self.u)
        weightUpdate = self.v
        return weightUpdate
    def update(self, weightUpdate):
        #import ipdb; ipdb.set_trace()
        self._setGradVec(weightUpdate * self._getLRVec())
        self._updateParamsWithGradVec()
        self.v = torch.zeros_like(self.v, device=self.device)
        return
    
    def forward(self, inputs, targets, training=True):
        self.sketchedModel.train(training)
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
        # weight decay
        if self.weight_decay != 0:
            gradVec.add_(self.weight_decay/self.num_workers, self._getParamVec())
        if self.nesterov:
            self.u.add_(gradVec).mul_(self.momentum)
            self.v.add_(self.u).add_(gradVec)
        else:
            self.u.mul_(self.momentum).add_(gradVec)
            self.v += (self.u)
        # this is v
        self.candidateSketch = self.v[self.sketchMask]
        self.sketch.zero()
        self.sketch += self.candidateSketch
        # COMMUNICATE ONLY THE TABLE
        return self.sketch.table

    def send_topkAndUnsketched(self, hhcoords):
    #    hhcoords = hhcoords.to(self.device)
        # directly send whatever wasn't sketched
        unsketched = self.v[~self.sketchMask]
        # COMMUNICATE
        return self.candidateSketch[hhcoords], unsketched
    #.cpu()
#     @ray.remote
    def apply_update(self, weightUpdate):
        # zero out the coords that are getting communicated
        self.u[weightUpdate.nonzero()] = 0
        self.v[weightUpdate.nonzero()] = 0
        self.v[~self.sketchMask] = 0
        self._setGradVec(weightUpdate * self._getLRVec())
        self._updateParamsWithGradVec()
        self.sketchedModel.zero_grad()
