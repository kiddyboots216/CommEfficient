import numpy as np
import torch
import torch.nn as nn
import ray

from sketcher import Sketcher
from worker import Worker
from core import warmup_cudnn
@ray.remote(
    num_gpus=0.5, 
    num_cpus=1.0
)
class ParameterServer(Sketcher):
    def __init__(self, model_maker, model_config, kwargs):
        print(f"Received args {kwargs}")
        self.step_number = 0
        self.params = kwargs
#         print(f"Now my params are {self.params}")
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
        
#     @ray.remote
    def compute_hhcoords(self, sketches):
#         from IPython.core.debugger import set_trace; set_trace()
        sketches = [sketch.to(self.device) for sketch in sketches]
        # THIS ON SERVER
#         sketches.append(self.sketch)
#         print(sketches)
#         for sketch in sketches:
#             self.sketch += sketch
        self.sketch += torch.sum(torch.stack(sketches, dim=0), dim=0)
#         for sketch in sketches:
#             self.sketch += sketch
#         candidateTopK = self.sketch.unSketch(k=self.p2*self.k)
        self.candidateTopK = self.sketch.unSketch(k=self.p2*self.k)
        self.candidateHHCoords = self.candidateTopK.nonzero()
        # don't need to stack or sum
        # COMMUNICATE
        return self.candidateHHCoords.cpu()
#     @ray.remote
    def compute_update(self, sketchesAndUnsketched):
#         from IPython.core.debugger import set_trace; set_trace()
        sketches, unsketched = sketchesAndUnsketched
        sketches = [sketch.to(self.device) for sketch in sketches]
        unsketched = [unsketch.to(self.device) for unsketch in unsketched]
        self.candidateTopK[self.candidateHHCoords] = torch.sum(
            torch.stack(sketches),dim=0)
#         del vs
#         del candidateSketch
        # this is w
        del self.candidateHHCoords
        weights = self.topk(self.candidateTopK, k=self.k)
        del self.candidateTopK
        weightUpdate = torch.zeros(self.grad_size, device=self.device)
#         weightUpdate = torch.zeros_like(self.vs[0])
        weightUpdate[self.sketchMask] = weights
        weightUpdate[~self.sketchMask] = torch.sum(torch.stack(unsketched), dim=0)
        # COMMUNICATE
        return weightUpdate.cpu()
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
