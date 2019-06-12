import torch
import torch.nn as nn
import ray

from csvec import CSVec
from sketched_model import SketchedModel

@ray.remote(num_gpus=1.0)
class Worker(Sketcher):
    def __init__(self, worker_index, model_maker, model_config, k, p2, numCols, numRows, lr,
                 momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 numBlocks=1, p1=0, step_number=0):
        self.worker_index = worker_index
        self.lr = lr
        trainable_params = lambda model: filter(lambda p: p.requires_grad, model.parameters())
        self.step_number = step_number
        model = model_maker(model_config)
        self.sketchedModel = SketchedModel(model)
        params = trainable_params(self.sketchedModel)
        super().__init__(params, k, p2, numCols, numRows, lr,
                 momentum, dampening, weight_decay, nesterov,
                 numBlocks, p1)
        self.u = torch.zeros(self.grad_size, device=self.device)
        self.v = torch.zeros(self.grad_size, device=self.device)
        self.sketch = CSVec(d=self.sketchMask.sum().item(), c=numCols, r=numRows, 
                            device=self.device, nChunks=1, numBlocks=numBlocks)
        
#     @ray.remote
    def forward(self, x, y):
        self.zero_grad()
        criterion = torch.nn.MSELoss(reduction='sum')
        loss = criterion(self.sketchedModel(x), y)
#         return loss
        loss.backward()
        return self.compute_sketch()
    
    def param_values(self):
            return {"lr": self.lr(self.step_number) if callable(self.lr) else self.lr}
    
    def compute_sketch(self):
        """
        Calls _backwardWorker inside backward
        """
        
        self.step_number += 1
        self.param_groups[0].update(**self.param_values())
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
        self.candidateSketch = self.v[self.sketchMask]
        self.sketch.zero()
        self.sketch += self.candidateSketch
        # COMMUNICATE ONLY THE TABLE
#         print(self.sketch.table.size())
        return self.sketch.table
#     @ray.remote
    def send_topkAndUnsketched(self, hhcoords):
        # directly send whatever wasn't sketched
        unsketched = self.v[~self.sketchMask]
        # COMMUNICATE
        return self.candidateSketch[hhcoords], unsketched
#     @ray.remote
    def apply_update(self, weightUpdate):
#         assert False
        # zero out the coords that are getting communicated
        self.u[weightUpdate.nonzero()] = 0
        self.v[weightUpdate.nonzero()] = 0
        self.v[~self.sketchMask] = 0
        """
        Return from _aggAndZeroSketched, finish _aggregateAndZeroUVs
        """
        """
        Return from _aggregateAndZeroUVs, back in backward
        """
        self._setGradVec(weightUpdate * self._getLRVec())
        self._updateParamsWithGradVec()