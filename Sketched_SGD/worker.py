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
    def __init__(self, num_workers, worker_index, model_maker, model_config, kwargs):
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.step_number = 0
        self.params = kwargs
        print(f"Initializing worker {worker_index}")
        super().__init__(model_maker, model_config, **self.param_values())
        warmed_up = False
        while not warmed_up:
            try:
                for size in [512, 256]:
                        warmup_cudnn(self.sketchedModel, size)
                warmed_up = True
            except RuntimeError as e:
                print(e)
#         k, p2, numCols, numRows, lr,
#                  momentum=0, dampening=0, weight_decay=0, nesterov=False,
#                  numBlocks=1, p1=0, step_number=0):
        
        # get params and set param_groups, self.sketch via Sketcher
#         trainable_params = lambda model: filter(lambda p: p.requires_grad, model.parameters())        
#         model = model_maker(model_config)
#         self.sketchedModel = SketchedModel(model)
#         params = trainable_params(self.sketchedModel)
#         super().__init__(model_maker, model_config, **)
        # make u, v, sketch
        self.u = torch.zeros(self.grad_size, device=self.device)
        self.v = torch.zeros(self.grad_size, device=self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.correctCriterion = Correct()
#     def param_values(self):
#         return {k: v(self.step_number) if callable(v) else v
#                 for k,v in self.params.items()}    
#     @ray.remote
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
        loss.sum().backward()
        """
        return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()
        if training: 
            loss.sum().backward()
            return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy(), self.compute_sketch() 
        else:
            return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy(), "bleh" 
        """
        if training:
         #   loss.sum().backward()
#             return self.compute_sketch()
            return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy(), self.compute_sketch()
        else:
#             pass
            return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy(), "bleh"
        #"""
    
    def compute_sketch(self): 
        """
        Calls _backwardWorker inside backward
        """
        
        self.step_number += 1
        self.param_groups[0].update(**self.param_values())
        gradVec = self._getGradVec()
        #self.v = gradVec
        # weight decay
        if self.weight_decay != 0:
            gradVec.add_(self.weight_decay/self.num_workers, self._getParamVec())
        # TODO: Pretty sure this momentum/residual formula is wrong
        if self.nesterov:
            self.u.add_(gradVec).mul_(self.momentum)
            self.v.add_(self.u).add_(gradVec)
#       #      self.us[0].add_(gradVec).mul_(self.momentum)
#       #      self.vs[0].add_(self.us[0]).add_(gradVec)
        else:
            self.u.mul_(self.momentum).add_(gradVec)
            self.v.add_(self.u)
#             self.us[0].mul_(self.momentum).add_(gradVec)
#             self.vs[0].add_(self.us[0])
        # this is v
#         vs = [v[self.sketchMask] for v in self.vs]
        self.candidateSketch = self.v[self.sketchMask]
        self.sketch.zero()
        self.sketch += self.candidateSketch
        # COMMUNICATE ONLY THE TABLE
#         print(self.sketch.table.size())
       # import pdb; pdb.set_trace()
        return self.sketch.table
    #.cpu()
#     @ray.remote
    def send_topkAndUnsketched(self, hhcoords):
    #    hhcoords = hhcoords.to(self.device)
        # directly send whatever wasn't sketched
        unsketched = self.v[~self.sketchMask]
        # COMMUNICATE
        return self.candidateSketch[hhcoords], unsketched
    #.cpu()
#     @ray.remote
    def apply_update(self, weightUpdate):
        #weightUpdate = weightUpdate.to(self.device)
#         assert False
        # zero out the coords that are getting communicated
        self.u[weightUpdate.nonzero()] = 0
        self.v[weightUpdate.nonzero()] = 0
        self.v[~self.sketchMask] = 0
        self._setGradVec(weightUpdate * self._getLRVec())
        self._updateParamsWithGradVec()
        self.sketchedModel.zero_grad()
    def apply_grad(self, weightUpdate): 
        self.u = torch.zeros_like(self.u, device=self.device)
        self.v = torch.zeros_like(self.v, device=self.device)
        self._setGradVec(weightUpdate * self._getLRVec())
        self._updateParamsWithGradVec()
