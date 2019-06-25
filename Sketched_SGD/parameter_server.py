import numpy as np
import torch
import torch.nn as nn
import ray

import line_profiler
profile = line_profiler.LineProfiler()
import atexit
atexit.register(profile.print_stats)

from sketcher import Sketcher
from worker import Worker
from core import warmup_cudnn

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

@ray.remote(
    num_gpus=2.0, 
    num_cpus=2.0
)
class ParameterServer(Sketcher):
    def __init__(self, model_maker, model_config, kwargs):
        print(f"Received args {kwargs}")
        self.step_number = 0
        self.params = kwargs
        super().__init__(model_maker, model_config, **self.param_values())
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
