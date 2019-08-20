import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from CommEfficient.minimal import CSVec

class SketchedModel:
    def __init__(self, model_cls, model_config, workers, 
                sketch_biases=False, sketch_params_larger_than=0):
        self.workers = np.array(workers)
        self.model = model_cls(**model_config)
        ray.wait([worker.set_model.remote(model_cls, model_config, 
            sketch_biases, sketch_params_larger_than) for worker in self.workers])

    def __call__(self, *args, **kwargs):
        input_minibatches = []
        batch_size = len(args[0])
        num_workers = len(self.workers)
        for i, _ in enumerate(self.workers):
            start = i * batch_size // num_workers
            end = (i+1) * batch_size // num_workers
            input_minibatches.append(args[0][start:end])
        return [worker.model_call.remote(
            input_minibatches[worker_id]) for worker_id, worker in enumerate(self.workers)]

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        if name in ["model", "workers"]:
            self.__dict__[name] = value
        else:
            [worker.model_setattr.remote(name, value) for worker in self.workers]

class SketchedLoss(object):
    def __init__(self, criterion, workers):
        self.workers = np.array(workers)
        [worker.set_loss.remote(criterion) for worker in self.workers]

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0:
            print("Kwargs aren't supported by Ray")
            return
        input_minibatches = args[0]
        target_minibatches = []
        batch_size = len(args[1])
        num_workers = len(self.workers)
        for i, _ in enumerate(self.workers):
            start = i * batch_size // num_workers
            end = (i+1) * batch_size // num_workers
            target_minibatches.append(args[1][start:end])
        return self._loss(input_minibatches, target_minibatches, self.workers)

    def _loss(self, input_minibatches, target_minibatches, workers):
        results = torch.stack(
             ray.get(
                 [worker.loss_call.remote(
                    input_minibatches[worker_id], target_minibatches[worker_id])
                 for worker_id, worker in enumerate(workers)]
             ), 
             dim=0)
        result = SketchedLossResult(results, workers)
        return result
    
class SketchedLossResult(object):
    def __init__(self, tensor, workers):
        #self._tensor = tensor.detach().cpu().numpy()
        #import pdb; pdb.set_trace()
        self._tensor = tensor
        self.workers = workers

    def backward(self):
        ray.wait([worker.loss_backward.remote()
            for worker in self.workers])

    def __repr__(self):
        return self._tensor.__repr__()

    def __getattr__(self, name):
        #if name=="mean":
        #    return lambda: 1
        return getattr(self._tensor, name)

class SketchedOptimizer(optim.Optimizer):
    def __init__(self, optimizer, workers):
        """
        Takes in an already-initialized optimizer and list of workers (object IDs).
        Gives the workers the optimizers and then wraps optimizer methods. 
        """
        self.workers = np.array(workers)
        self.head_worker = self.workers[0]
        # self.param_groups = optimizer.param_groups
        ray.wait([worker.set_optimizer.remote(optimizer) for worker in self.workers])
    
    def zero_grad(self):
        self._zero_grad(self.workers)

    def _zero_grad(self, workers):
        [worker.optimizer_zero_grad.remote() for worker in workers]

    def step(self):
        self._step(self.workers, self.workers)

    def _step(self, train_workers, update_workers):
        grads = [worker.compute_grad.remote() for worker in train_workers]
        ray.wait([worker.all_reduce_sketched.remote(*grads) for worker in update_workers]) 

    def __getattr__(self, name):
        if name=="param_groups":
            param_groups = ray.get(self.head_worker.get_param_groups.remote())
            #print(f"Param groups are {param_groups}")
            return [SketchedParamGroup(param_group, self.workers, idx
                ) for idx, param_group in enumerate(param_groups)]

class SketchedParamGroup(object):
    def __init__(self, param_group, workers, index):
        """
        workers is a list of object IDs
        """
        self.workers = workers
        self.param_group = param_group
        self.index = index

    def setdefault(self, name, value):
        ray.wait([worker.param_group_setdefault.remote(self.index, name, value) for worker in self.workers])
    
    def __getitem__(self, name):
        return self.param_group.__getitem__(name)
    
    def __setitem__(self, name, value):
        ray.wait([worker.param_group_setitem.remote(self.index, name, value) for worker in self.workers])

@ray.remote(num_gpus=1)
class SketchedWorker(object):
    def __init__(self, args,
                sketch_params_larger_than=0, sketch_biases=False):
        self.num_workers = args['num_workers']
        self.k = args['k']
        self.p2 = args['p2']
        self.num_cols = args['num_cols']
        self.num_rows = args['num_rows']
        self.num_blocks = args['num_blocks']
        self.lr = args['lr']
        self.momentum = args['momentum']
        self.dampening = args['dampening']
        self.weight_decay = args['weight_decay']
        self.nesterov = args['nesterov']
        self.sketch_params_larger_than = sketch_params_larger_than
        self.sketch_biases = sketch_biases
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_model(self, model_cls, model_config, 
            sketch_biases, sketch_params_larger_than):
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        model = model_cls(**model_config)
        torch.random.set_rng_state(rand_state)
        for p in model.parameters():
            p.do_sketching = p.numel() >= sketch_params_larger_than
        # override bias terms with whatever sketchBiases is
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                if m.bias is not None:
                    m.bias.do_sketching = sketch_biases
        self.model = model.to(self.device)

    def set_loss(self, criterion):
        self.criterion = criterion.to(self.device)

    def model_call(self, *args):
        #import pdb; pdb.set_trace()
        args = [arg.to(self.device) for arg in args]
        self.outs = self.model(*args)
        return self.outs

    def model_getattr(self, name):
        return getattr(self.model, name)

    def model_setattr(self, name, value):
        if name == "model":
            self.__dict__[name] = value
        else:
            self.model.setattr(name, value)

    def param_group_setitem(self, index, name, value):
        self.param_groups[index].__setitem__(name, value)
        
    def param_group_setattr(self, index, name, value):
        self.param_groups[index].setattr(name, value)
        
    def param_group_setdefault(self, index, name, value):
        self.param_groups[index].setdefault(name, value)
        
    def get_param_groups(self):
        try:
            return [{'initial_lr': group['initial_lr'], 'lr': group['lr']} for group in self.param_groups]
        except Exception as e:
            #print(f"Exception is {e}")
            return [{'lr': group['lr']} for group in self.param_groups]
    
    def loss_call(self, *args):
        args = [arg.to(self.device) for arg in args]
        self.loss = self.criterion(self.outs, args[1])
        return self.loss

    def loss_backward(self):
        #import pdb; pdb.set_trace()
        self.loss.sum().backward()
        #del self.outs

    def set_optimizer(self, opt):
        assert self.model is not None, "model must be already initialized"
        p = opt.param_groups[0]
        lr = p['lr']
        dampening = p['dampening']
        nesterov = p['nesterov']
        weight_decay = p['weight_decay']
        momentum = p['momentum']
        opt = optim.SGD(self.model.parameters(), 
            lr=lr, 
            dampening=dampening, 
            nesterov=nesterov, 
            weight_decay=weight_decay, 
            momentum=momentum)
        self.param_groups = opt.param_groups
        grad_size = 0
        sketch_mask = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    size = torch.numel(p)
                    if p.do_sketching:
                        sketch_mask.append(torch.ones(size))
                    else:
                        sketch_mask.append(torch.zeros(size))
                    grad_size += size
        self.grad_size = grad_size
        print(f"Total dimension is {self.grad_size} using k {self.k} and p2 {self.p2}")
        self.sketch_mask = torch.cat(sketch_mask).byte().to(self.device)
        print(f"sketch_mask.sum(): {self.sketch_mask.sum()}")
        self.sketch = CSVec(d=self.sketch_mask.sum().item(), 
            c=self.num_cols,
            r=self.num_rows,
            device=self.device,
            nChunks=1,
            numBlocks=self.num_blocks)
        self.u = torch.zeros(self.grad_size, device=self.device)
        self.v = torch.zeros(self.grad_size, device=self.device)

    def optimizer_zero_grad(self):
        self._zero_grad()

    def compute_grad(self):
        #assert self._getLRVec() != 0.0, "invalid lr"
        # compute grad 
        gradVec = self._getGradVec()
        #return gradVec
        # weight decay
        if self.weight_decay != 0:
            gradVec.add_(self.weight_decay/self.num_workers, self._getParamVec())
        if self.nesterov:
            #import pdb; pdb.set_trace()
            self.u.add_(gradVec).mul_(self.momentum)
            self.v.add_(self.u).add_(gradVec)
        else:
            self.u.mul_(self.momentum).add_(gradVec)
            self.v += (self.u)
            #self.v = gradVec
        # this is v
        return self.v * self._getLRVec()
        candidateSketch = self.v[self.sketchMask]
        self.sketch.zero()
        self.sketch += candidateSketch
        del candidateSketch
        # COMMUNICATE ONLY THE TABLE
        return self.sketch.table

    def all_reduce_sketched(self, *grads):
        # compute update
        """
        grads = [grad.to(self.device) for grad in grads]
        self._apply_update(torch.mean(torch.stack(grads), dim=0))
        return
        """
        self.sketch.zero()
        for grad in grads:
            self.sketch += grad[self.sketch_mask]
        candidate_top_k = self.sketch.unSketch(k=self.p2*self.k)
        candidate_hh_coords = candidate_top_k.nonzero()
        hhs = [grad[candidate_hh_coords] for grad in grads]
        candidate_top_k[candidate_hh_coords] = torch.sum(
            torch.stack(hhs),dim=0)
        weights = self._topk(candidate_top_k, k=self.k)
        weight_update = torch.zeros(self.grad_size, device=self.device)
        weight_update[self.sketch_mask] = weights
        weight_update[~self.sketch_mask] = torch.sum(
            torch.stack(
                [grad[~self.sketch_mask] for grad in grads]), dim=0)
        self._apply_update(weight_update)
        #"""

    def _topk(self, vec, k):
        """ Return the largest k elements (by magnitude) of vec"""
        ret = torch.zeros_like(vec)
        # on a gpu, sorting is faster than pytorch's topk method
        topkIndices = torch.sort(vec**2)[1][-k:]
        ret[topkIndices] = vec[topkIndices]
        return ret

    def _apply_update(self, update):
        # set update
        self.u[update.nonzero()] = 0
        self.v[update.nonzero()] = 0
        self.v[~self.sketch_mask] = 0
        #self.sync(weightUpdate * self._getLRVec())
        weight_update = update
        # weight_update = update * self._getLRVec()
        #import pdb; pdb.set_trace()
        weight_update = weight_update.to(self.device)
        start = 0
        for param_group in self.param_groups:
            for p in param_group['params']:
                end = start + torch.numel(p)
                # we previously had diff_vec = copy - (copy - grad) = grad, so subtract here 
                p.data.add_(-weight_update[start:end].reshape(p.data.shape))
                start = end
        #import pdb; pdb.set_trace()
        # self._setGradVec(weight_update)
        # self._updateParamsWithGradVec()

    def _getLRVec(self):
        """Return a vector of each gradient element's learning rate
        If all parameters have the same learning rate, this just
        returns torch.ones(D) * learning_rate. In this case, this
        function is memory-optimized by returning just a single
        number.
        """
        if len(self.param_groups) == 1:
            lr = self.param_groups[0]["lr"]
#            print(f"Lr is {lr}")
            return lr

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

    def _zero_grad(self):
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
        #import pdb; pdb.set_trace()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(-p.grad.data)
