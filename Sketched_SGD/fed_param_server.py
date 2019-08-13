import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from minimal import CSVec

from sketched_classes import SketchedLossResult, SketchedParamGroup

@ray.remote(num_gpus=1)
class FedParamServer:
    def __init__(self, args,
    	sketch_params_larger_than=0, sketch_biases=False):
        self.k = args['k']
        self.p2 = args['p2']
        self.num_cols = args['num_cols']
        self.num_rows = args['num_rows']
        self.num_blocks = args['num_blocks']
        self.rounds = []
        self.sketch_params_larger_than = sketch_params_larger_than
        self.sketch_biases = sketch_biases
        self.device = torch.device("cuda" if 
            torch.cuda.is_available() else "cpu")

    def sync(self, round_id):
        return self.rounds[-1]
        stale_update = self.rounds[round_id]
        return stale_update

    def all_reduce_sketched(self, *grads):
        # compute update
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

    def _apply_update(self, update):
        curr_weights = self.rounds[-1]
        updated_weights = curr_weights - (update * self._getLRVec())
        #import pdb; pdb.set_trace()
        self.rounds.append(updated_weights)
        self._sync_weights(updated_weights)

    def _sync_weights(self, update):
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
                p.data = update[startPos:startPos + size].reshape(shape)
                startPos += size
        #self.sync(weightUpdate * self._getLRVec())
        # weight_update = update * self._getLRVec()
        # #import pdb; pdb.set_trace()
        # weight_update = weight_update.to(self.device)
        # start = 0
        # for param_group in self.param_groups:
        #     for p in param_group['params']:
        #         end = start + torch.numel(p)
        #         p.data.add_(-weight_update[start:end].reshape(p.data.shape))
        #         start = end
        #import pdb; pdb.set_trace()
        # self._setGradVec(weight_update)
        # self._updateParamsWithGradVec()

    def param_group_setitem(self, index, name, value):
        self.param_groups[index].__setitem__(name, value)
        
    def param_group_setattr(self, index, name, value):
        self.param_groups[index].setattr(name, value)
        
    def param_group_setdefault(self, index, name, value):
        self.param_groups[index].setdefault(name, value)
        
    def _topk(self, vec, k):
        """ Return the largest k elements (by magnitude) of vec"""
        ret = torch.zeros_like(vec)
        # on a gpu, sorting is faster than pytorch's topk method
        topkIndices = torch.sort(vec**2)[1][-k:]
        ret[topkIndices] = vec[topkIndices]
        return ret

    def set_optimizer(self, opt_param_groups):
        assert self.model is not None, \
        "model must be already initialized"
        p = opt_param_groups[0]
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
        # del self.model
        self.param_groups = opt.param_groups
        self.grad_size = 0
        sketch_mask = []
        weight_vec = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    size = torch.numel(p)
                    if p.do_sketching:
                        sketch_mask.append(torch.ones(size))
                    else:
                        sketch_mask.append(torch.zeros(size))
                    weight_vec.append(p.data.view(-1).float())
                    d = p.data.view(-1).float()
                    self.grad_size += size
        # del self.param_groups
        weight_vec = torch.cat(weight_vec).float().to(self.device) 
        self.rounds.append(weight_vec)
        self.sketch_mask = torch.cat(sketch_mask).byte().to(self.device)
        self.sketch = CSVec(d=self.sketch_mask.sum().item(), 
            c=self.num_cols,
            r=self.num_rows,
            device=self.device,
            nChunks=1,
            numBlocks=self.num_blocks)
        print(f"Total dimension is {self.grad_size} using k {self.k} and p2 {self.p2} with sketch_mask.sum(): {self.sketch_mask.sum()}")

    def set_model(self, model_cls, model_config, 
            sketch_biases, sketch_params_larger_than):
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        model = model_cls(**model_config).to(self.device)
        torch.random.set_rng_state(rand_state)
        for p in model.parameters():
            p.do_sketching = p.numel() >= sketch_params_larger_than
        # override bias terms with whatever sketchBiases is
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                if m.bias is not None:
                    m.bias.do_sketching = sketch_biases
        self.model = model

    def model_call(self, *args):
        args = [arg.to(self.device) for arg in args]
        self.outs = self.model(*args)
        return self.outs
        #self.cuda()
        outs = self.model(args[0].to(self.device))
        #print(f"Length of self.outs is {len(self.outs)}")
        return outs

    def set_loss(self, criterion):
        self.criterion = criterion.to(self.device)

    def loss_call(self, *args):
        args = [arg.to(self.device) for arg in args[:2]]
        self.loss = self.criterion(self.outs, args[1])
        return self.loss
        loss = self.criterion(
            args[0].to(self.device), 
            args[1].to(self.device))
        #self.cpu()
        return loss

    def get_param_groups(self):
        try:
            return [{'initial_lr': group['initial_lr'],
             'lr': group['lr']} for group in self.param_groups]
        except Exception as e:
            #print(f"Exception is {e}")
            return [{'lr': group['lr']} for group in self.param_groups]

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
