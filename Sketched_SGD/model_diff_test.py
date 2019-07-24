import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SketchedModel:
    # not inheriting from nn.Module to avoid the fact that implementing
    # __getattr__ on a nn.Module is tricky, since self.model = model
    # doesn't actually add "model" to self.__dict__ -- instead, nn.Module
    # creates a key/value pair in some internal dictionary that keeps
    # track of submodules
    def __init__(self, model, workers, sketchBiases=False, sketchParamsLargerThan=0):
        self.model = model
        self.workers = workers
        # sketch everything larger than sketchParamsLargerThan
        for p in self.model.parameters():
            p.do_sketching = p.numel() >= sketchParamsLargerThan

        # override bias terms with whatever sketchBiases is
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                if m.bias is not None:
                    m.bias.do_sketching = sketchBiases
#         [worker.set_model.remote(self.model) for worker in self.workers]
        [worker.set_model(self.model) for worker in self.workers]

    def __call__(self, *args, **kwargs):
#         return [worker.model_call.remote(*args) for worker in self.workers]
        return [worker.model_call(*args) for worker in self.workers]

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        if name in ["model", "workers"]:
            self.__dict__[name] = value
        else:
            self.model.setattr(name, value)

class SketchedLoss(object):
    def __init__(self, criterion, workers):
        self.workers = workers
        [worker.set_loss.remote(criterion) for worker in self.workers]

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0:
            print("Kwargs aren't supported by Ray")
            return
        # TODO: fix this partitioning
        results = [worker.loss_call.remote(args)
                    for worker_id, worker in enumerate(self.workers)]
#         for worker_id, worker in enumerate(self.workers):
#             worker.loss_call.remote(args[worker_id])
#         [worker.loss_call.remote()
#         for worker_id, worker in enumerate(self.workers)]
#         results = torch.zeros(2)
#         results = torch.stack(
#             *ray.get(
#                 [worker.loss_call.remote(args[worker_id])
#                 for worker_id, worker in enumerate(self.workers)]
#             ), 
#             dim=0)
        results.register_backward_hook(self._backward)

    def _backward(self):
        [worker.loss_backward.remote()
            for worker in self.workers]

class SketchedOptimizer(object):
    def __init__(self, optimizer, workers):
        """
        Takes in an already-initialized optimizer and list of workers (object IDs).
        Gives the workers the optimizers and then wraps optimizer methods. 
        """
        [worker.set_optimizer.remote(optimizer) for worker in workers]
    
    def zero_grad(self):
        [worker.optimizer_zero_grad.remote() for worker in workers]

    def step(self):
        grads = [worker.compute_grad.remote() for worker in workers]
        [worker.all_reduce_sketched.remote(*grads) for worker in workers]

# @ray.remote(num_gpus=1)
class SketchedWorker(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_model(self, model):
        self.model = model.to(self.device)

    def set_loss(self, criterion):
        self.loss = criterion.to(self.device)

    def model_call(self, *args):
        return self.model(*args)

    def model_getattr(self, name):
        return getattr(self.model, name)

    def model_setattr(self, name, value):
        if name == "model":
            self.__dict__[name] = value
        else:
            self.model.setattr(name, value)

    def loss_call(self, *args):
        return self.loss(*args)

    def loss_backward(self):
        self.loss.backward()

    def set_optimizer(self, optimizer):
        self.opt = optimizer
        grad_size = 0
        sketchMask = []
        for group in self.opt.param_groups:
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
        self.u = torch.zeros(self.grad_size, device=self.device)
        self.v = torch.zeros(self.grad_size, device=self.device)

    def optimizer_zero_grad(self):
        self.opt.zero_grad()

    def compute_grad(self):
        # compute grad 
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
            #self.v = gradVec
        # this is v
        return self.v
        candidateSketch = self.v[self.sketchMask]
        self.sketch.zero()
        self.sketch += candidateSketch
        del candidateSketch
        # COMMUNICATE ONLY THE TABLE
        return self.sketch.table
        return grad

    def all_reduce_sketched(self, *grads):
        # compute update
        grads = [grad.to(self.device) for grad in grads]
        self.sketch.zero()
        for grad in grads:
            self.sketch += grad[self.sketch_mask]
            #/len(diff_vecs)
        candidate_top_k = self.sketch.unSketch(k=self.p2*self.k)
        candidate_hh_coords = candidate_top_k.nonzero()
        hhs = [diff_vec[candidate_hh_coords] for diff_vec in diff_vecs]
        candidate_top_k[candidate_hh_coords] = torch.sum(
            torch.stack(hhs),dim=0)
        weights = self.topk(candidate_top_k, k=self.k)
        weight_update = torch.zeros(self.grad_size, device=self.device)
        weight_update[self.sketch_mask] = weights
        weight_update[~self.sketch_mask] = torch.sum(
            torch.stack(
                [diff_vec[~self.sketch_mask] for diff_vec in diff_vecs]), dim=0)
        self.apply_update(weight_update)

    def _apply_update(self, update):
        # set update
        self.u[weightUpdate.nonzero()] = 0
        self.v[weightUpdate.nonzero()] = 0
        self.v[~self.sketchMask] = 0
        #self.sync(weightUpdate * self._getLRVec())
        self._setGradVec(weightUpdate * self._getLRVec())
        self._updateParamsWithGradVec()

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
    def sync(self, vec):
        """Set params"""
        gradShapes, gradSizes = self._getGradShapes()
        startPos = 0
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                shape = gradShapes[i]
                size = gradSizes[i]
                i += 1
                assert(size == torch.numel(p))
                p.data = vec[startPos:startPos + size].reshape(shape)
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

ray.init(ignore_reinit_error=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CONSTANTS
epochs = 1
batch_size = 1
D_in, D_out, H_sizes = 2, 4, [2,4]

x = torch.randn(batch_size, D_in, device=device)
y = torch.randn(batch_size, D_out, device=device)
num_workers = 2


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

model = FCNet(**model_config).cuda()
workers = [SketchedWorker() for _ in range(2)]
sketched_model = SketchedModel(model, workers)
batch_size = 32
x = torch.randn(batch_size, D_in, device="cuda")
y = torch.randn(batch_size, D_out, device="cuda")
outputs = sketched_model(x)
criterion = torch.nn.MSELoss(reduction='sum')
sketched_criterion = SketchedLoss(criterion, workers)
sketched_criterion(outputs, y)
# train_loss = sketched_criterion(sketched_model(x), y)
# train_loss.backward()
# sketched_optim.step()