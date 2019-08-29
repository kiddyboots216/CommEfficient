import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from csvec import CSVec

from CommEfficient.sketched_classes import SketchedLossResult, SketchedParamGroup

@ray.remote(num_gpus=0.5)
class FedParamServer:
    def __init__(self, args,
    	sketch_params_larger_than=0, sketch_biases=False):
        self.k = args['k']
        self.p2 = args['p2']
        self.num_cols = args['num_cols']
        self.num_rows = args['num_rows']
        self.num_blocks = args['num_blocks']
        self.momentum = args['momentum']
        self.nesterov = args['nesterov']
        self.n_clients_per_round = args['n_clients_per_round']
        self.rounds = []
        self.sketch_params_larger_than = sketch_params_larger_than
        self.sketch_biases = sketch_biases
        self.device = torch.device("cuda" if
            torch.cuda.is_available() else "cpu")
        print(f"Creating param server")

    def sketch_latest(self, round_id):
        diff_vec = self.get_latest(round_id)
        return diff_vec
        self.sketch.zero()
        self.sketch.accumulateVec(diff_vec)

    def sync(self, round_id):
        w_stale = self.rounds[round_id].to(self.device)
        w_new = self.rounds[-1].to(self.device)
        desired_diff = w_new - w_stale
        # find a datum such that the gradient of the model evaluated
        # on that datum using w_stale equals desired_diff
        #raise RuntimeWarning("overriding your model's requires_grad " + \
        #                     "due to extreme laziness")
        def get_param_vec(model):
            param_vec = []
            start = 0
            for p in model.parameters():
                end = start + p.numel()
                param_vec.append(p.data.view(-1))
                start = end
            return torch.cat(param_vec).to(self.device)

        def set_param_vec(model, param_vec):
            start = 0
            for p in model.parameters():
                end = start + p.numel()
                p.data.zero_()
                p.data.add_(param_vec[start:end].view(p.size()))
                start = end

        def get_grad_vec(model):
            start = 0
            grad_vec = []
            for p in model.parameters():
                end = start + p.numel()
                if p.grad is None:
                    grad_vec.append(torch.zeros(p.numel()))
                else:
                    grad_vec.append(p.grad.view(-1))
                start = end
            return torch.cat(grad_vec).to(self.device)

        def set_grad_vec(model, grad_vec):
            start = 0
            for p in model.parameters():
                end = start + p.numel()
                p.grad.data.zero_()
                p.data.add_(grad_vec[start:end].view(p.grad.size()))
                start = end

        orig_param_vec = get_param_vec(self.model)
        set_param_vec(self.model, w_stale)
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)

        d = torch.randn(1, 3, 32, 32).to(self.device)
        torch.random.set_rng_state(rand_state)
        d.requires_grad = True

        # all workers know to use class 0 as the target
        fake_target = torch.tensor([0]).long().to(self.device)

        # save the original gradient
        orig_grad = get_grad_vec(self.model)

        # minimize ||model_grad - desired_diff||_2^2 over d
        opt = torch.optim.SGD((d,), lr=0.01, momentum=0.9)
        for i in range(100):
            opt.zero_grad()
            self.model.zero_grad()
            fake_loss = self.criterion(self.model(d), fake_target)
            fake_loss.backward(create_graph=True)
            model_grad = get_grad_vec(self.model)

            loss = torch.sum((model_grad - desired_diff)**2)
            data_grad = torch.autograd.grad(loss, d, only_inputs=True)[0]

            opt.param_groups[0]["params"][0].grad = data_grad
            opt.step()

        # reset to original state
        set_grad_vec(self.model, orig_grad)
        set_param_vec(self.model, orig_param_vec)
        return desired_diff, d

    def get_latest(self, round_id):
        stale_weights = self.rounds[round_id].to(self.device)
        curr_weights = self.rounds[-1].to(self.device)
        diff_vec = torch.sub(curr_weights, stale_weights)
        return diff_vec.detach().cpu()

    def _virtual_momentum(self, gradVec):
        return gradVec
        if self.nesterov:
            self.u.add_(gradVec).mul_(self.momentum)
            self.v.add_(self.u).add_(gradVec)
        else:
            self.u.mul_(self.momentum).add_(gradVec)
            #self.v += (self.u)
        return self.u

    def server_update(self, *grads):
        grads = [grad.to(self.device) for grad in grads]
        update = self._all_reduce_sketched(*grads)
        weight_update = self._virtual_momentum(update)
        self._apply_update(weight_update)

    def _all_reduce_sketched(self, *grads):
        self.sketch.zero()
        for grad in grads:
            self.sketch.accumulateVec(grad[self.sketch_mask])
        if self.p2 > 0:
            candidate_top_k = self.sketch.unSketch(
                    k=self.p2*self.k)
            candidate_hh_coords = candidate_top_k.nonzero()
            hhs = [grad[candidate_hh_coords] for grad in grads]
            candidate_top_k[candidate_hh_coords] = sum(hhs)
            weights = self._topk(candidate_top_k, k=self.k)
        else:
            weights = self.sketch.unSketch(k=self.k)
        weight_update = torch.zeros(self.grad_size, device=self.device)
        weight_update[self.sketch_mask] = weights
        weight_update[~self.sketch_mask] = sum(
            [grad[~self.sketch_mask] for grad in grads])
        return weight_update

    def _update_vecs(self, weight_update):
        self.u[weight_update.nonzero()] = 0
        self.v[weight_update.nonzero()] = 0
        self.v[~self.sketch_mask] = 0

    def _apply_update(self, update):
        curr_weights = self.rounds[-1].to(self.device)
        weight_update = update * self._getLRVec()
        updated_weights = curr_weights - weight_update
        self.rounds.append(updated_weights.cpu())
        start = 0
        for param_group in self.param_groups:
            for p in param_group['params']:
                end = start + torch.numel(p)
                p.data = updated_weights[start:end].reshape(p.data.shape)
                start = end

    def model_getattr(self, name):
        return getattr(self.model, name)

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
                    self.grad_size += size
        weight_vec = torch.cat(weight_vec).float().to(self.device)
        self.rounds.append(weight_vec.cpu())
        self.sketch_mask = torch.cat(sketch_mask).bool().to(self.device)
        self.sketch = CSVec(d=self.sketch_mask.sum().item(),
            c=self.num_cols,
            r=self.num_rows,
            device=self.device,
            numBlocks=self.num_blocks)
        #print(f"Total dimension is {self.grad_size} using k {self.k} and p2 {self.p2} with sketch_mask.sum(): {self.sketch_mask.sum()}")
        self.u = torch.zeros(self.grad_size, device=self.device)
        self.v = torch.zeros(self.grad_size, device=self.device)

    def set_model(self, model_cls, model_config,
            sketch_biases, sketch_params_larger_than):
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        model = model_cls(**model_config).to(self.device)
        torch.random.set_rng_state(rand_state)
        for p in model.parameters():
            size = p.numel()
            p.do_sketching = size >= sketch_params_larger_than
        # override bias terms with whatever sketchBiases is
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                if m.bias is not None:
                    m.bias.do_sketching = sketch_biases
        self.model = model

    def model_call(self, batch):
        self.outs = self.model(batch.to(self.device))
        return self.outs.cpu()

    def set_loss(self, criterion):
        self.criterion = criterion.to(self.device)

    def loss_call(self, *args):
        args = [arg.to(self.device) for arg in args[:2]]
        self.loss = self.criterion(self.outs, args[1])
        return self.loss.detach().cpu().numpy()
        loss = self.criterion(
            args[0].to(self.device),
            args[1].to(self.device))
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
