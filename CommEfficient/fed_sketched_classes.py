import ray
import torch
import torch.optim as optim
import numpy as np
from csvec import CSVec
from CommEfficient.sketched_classes import SketchedLossResult, SketchedParamGroup

class FedSketchedModel:
    def __init__(self, model_cls, model_config, workers,
                param_server, fed_params, device,
                sketch_biases=False, sketch_params_larger_than=0):
        self.cur_round = 0
        self.rounds = []
        self.workers = np.array(workers)
        self.param_server = param_server
        self.device = device
        to_set_model = np.append(self.workers, self.param_server)
        self.model = model_cls(**model_config)
        ray.wait([worker.set_model.remote(model_cls, model_config,
            sketch_biases, sketch_params_larger_than
            ) for worker in to_set_model])
        self.workers_last_updated = {i:0 for i in range(len(workers))}

    def train(self, training):
        self.training = training

    def __call__(self, worker_batches, worker_ids=None):
        if self.training:
            if worker_ids is None:
                raise ValueError("must supply worker_ids if training")
            participating_workers = self.workers[worker_ids]
            if self.cur_round > 0:
                futures = [w.sketched_update.remote(
                            self.param_server.get_latest.remote(
                               self.workers_last_updated[worker_ids[w_id]]
                            ), self.cur_round)
                           for w_id, w in enumerate(participating_workers)]
                ray.wait(futures)
                for w_id in range(len(participating_workers)):
                    self.workers_last_updated[worker_ids[w_id]] = self.cur_round
            outs = [w.model_call.remote(worker_batches[w_id].detach().cpu())
                    for w_id, w in enumerate(participating_workers)]
            return [o.to(self.device) for o in ray.get(outs)]
        else:
            return self.param_server.model_call.remote(worker_batches.detach().cpu())

    def __setattr__(self, name, value):
        if name in ["model", "workers", "participation", "device",
                    "rounds", "param_server", "training",
                    "cur_round", "workers_last_updated"]:
            self.__dict__[name] = value
        else:
            [worker.model_setattr.remote(
                name, value) for worker in self.workers]

    def __getattr__(self, name):
        return ray.get(self.param_server.model_getattr.remote(name))

class FedSketchedOptimizer(optim.Optimizer):
    def __init__(self, optimizer, workers, param_server, fed_model):
        self.workers = np.array(workers)
        self.param_server = param_server
        self.model = fed_model
        self.cur_round = 0
        to_set_optimize = np.append(self.workers, self.param_server)
        p = optimizer.param_groups[0]
        optimizer_param_groups = [
                {'lr': p['lr'], 'dampening': p['dampening'],
                    'nesterov': p['nesterov'], 'momentum': p['momentum'],
                    'weight_decay': p['weight_decay']
                    }
        ]
        ray.wait([worker.set_optimizer.remote(
            optimizer_param_groups) for worker in to_set_optimize])

    def step_no_sync(self):
        train_workers, update_workers = self._get_workers()
        self._step(train_workers, train_workers)

    def step(self, idx):
        workers = self.workers[idx]
        self.cur_round += 1
        self.model.cur_round = self.cur_round
        self._step(workers, self.param_server)

    def zero_grad(self, idx):
        workers = self.workers[idx]
        ray.wait(
        [worker.optimizer_zero_grad.remote() for worker in workers]
        )

    def _step(self, train_workers, param_server):
        grads = [worker.compute_grad.remote(
            ) for worker in train_workers]
        ray.wait(
        [param_server.server_update.remote(*grads)]
        )

    def _get_workers(self):
        cur_round = self.model.rounds[-1]
        participating_clients = self.workers[cur_round]
        return participating_clients, [self.param_server]

    def __getattr__(self, name):
        if name=="param_groups":
            param_groups = ray.get(
                self.param_server.get_param_groups.remote())
            return [SketchedParamGroup(
                    param_group, [self.param_server], idx
                    ) for idx, param_group in enumerate(param_groups)]

class FedSketchedLoss:
    def __init__(self, criterion, workers, param_server, fed_model):
        self.model = fed_model
        self.workers = np.array(workers)
        self.param_server = param_server
        to_set_loss = np.append(self.workers, self.param_server)
        ray.wait([worker.set_loss.remote(criterion) for worker in to_set_loss])

    def __call__(self, outs, target_minibatches, worker_indices=None, **kwargs):
        if len(kwargs) > 0:
            print("Kwargs aren't supported by Ray")
            return
        if worker_indices is None:
            # doing validation now
            outs = self.param_server.loss_call.remote(outs, target_minibatches.detach().cpu())
            results = ray.get(outs)
            #results = outs
            result = SketchedLossResult(results, [])
            return result
        else:
            workers = self.workers[worker_indices]
            return self._loss(outs, target_minibatches, workers)
            """
            target_minibatches = []
            batch_size = len(targets)
            n_clients = len(self.workers)
            for i, _ in enumerate(self.workers):
                start = i * batch_size // n_clients
                end = (i+1) * batch_size // n_clients
                target_minibatches.append(targets[start:end])
            return self._loss(outs, target_minibatches, workers)
            """

        if len(args) == 2:
            results = ray.get(self.param_server.loss_call.remote(*args))
            result = SketchedLossResult(results, [])
            return result
        else:
            #participating_clients = self._get_workers()
            participating_clients = self.workers
            results = torch.stack(
                 ray.get(
                     [worker.loss_call.remote() for worker_id,
                     worker in enumerate(participating_clients)]
                 ),
                 dim=0)
            result = SketchedLossResult(results, participating_clients)
            return result

    def _loss(self, input_minibatches, target_minibatches, workers):
        assert torch.cuda.is_available(), "Why isn't CUDA available?"
        outs = [worker.loss_call.remote(#input_minibatches[w_id],
                                        target_minibatches[w_id].detach().cpu())
                for w_id, worker in enumerate(workers)]
        #results = torch.stack(ray.get(outs), dim=0)
        results = np.stack(ray.get(outs), axis=0)
        #results = outs
        result = SketchedLossResult(results, workers)
        return result

    def _get_workers(self):
        cur_round = self.model.rounds[-1]
        #print(f"Cur round for loss: {cur_round}")
        participating_clients = self.workers[cur_round]
        return participating_clients

# note: when using FedSketchedWorker,
# many more GPUs than are actually available must be specified
@ray.remote(num_gpus=0.5)
class FedSketchedWorker(object):
    def __init__(self, args,
                sketch_params_larger_than=0, sketch_biases=False):
        self.n_clients = args['n_clients']
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
        self.device = torch.device("cuda" if
            torch.cuda.is_available() else "cpu")
        self.cur_round = 0
        self.outs = None
        print(f"Creating worker")

    def set_model(self, model_cls, model_config,
            sketch_biases, sketch_params_larger_than):
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        model = model_cls(**model_config).to(self.device)
        torch.random.set_rng_state(rand_state)
        for p in model.parameters():
            size = p.numel()
            p.do_sketching = size >= sketch_params_larger_than
            #p.data.zero_()
        # override bias terms with whatever sketchBiases is
        for m in model.modules():
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.do_sketching = sketch_biases
        self.model = model

    def set_loss(self, criterion):
        self.criterion = criterion.to(self.device)

    def get_last_round(self):
        return self.cur_round

    def model_call(self, input_data):
        self.outs = self.model(input_data.to(self.device))
        return self.outs.detach().cpu()

    def loss_call(self, targets):
        self.loss = self.criterion(self.outs, targets.to(self.device))
        return self.loss.detach().cpu().numpy()

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
            return [{'initial_lr': group['initial_lr'],
             'lr': group['lr']} for group in self.param_groups]
        except Exception as e:
            #print(f"Exception is {e}")
            return [{'lr': group['lr']} for group in self.param_groups]

    def loss_backward(self):
        #import pdb; pdb.set_trace()
        self._zero_grad()
        self.loss.sum().backward()
        del self.outs
        del self.loss

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
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    size = torch.numel(p)
                    if p.do_sketching:
                        sketch_mask.append(torch.ones(size))
                    else:
                        sketch_mask.append(torch.zeros(size))
                    self.grad_size += size
        self.sketch_mask = torch.cat(sketch_mask).bool().to(self.device)
        self.sketch = CSVec(d=self.sketch_mask.sum().item(),
            c=self.num_cols,
            r=self.num_rows,
            device=self.device,
            numBlocks=self.num_blocks)
        print(f"Total dimension is {self.grad_size} using k {self.k} and p2 {self.p2}  with sketch_mask.sum(): {self.sketch_mask.sum()}")
        self.u = torch.zeros(self.grad_size, device=self.device)
        self.v = torch.zeros(self.grad_size, device=self.device)

    def optimizer_zero_grad(self):
        self._zero_grad()

    def compute_grad(self):
        # compute grad
        gradVec = self._getGradVec().to(self.device)

        # weight decay
        if self.weight_decay != 0:
            gradVec.add_(self.weight_decay/self.n_clients,
                        self._getParamVec())
        self.v.add_(gradVec)
        return self.v.detach().cpu()
        if self.nesterov:
            self.u.add_(gradVec).mul_(self.momentum)
            self.v.add_(self.u).add_(gradVec)
        else:
            self.u.mul_(self.momentum).add_(gradVec)
            self.v += (self.u)
        return self.v

    def sketched_update(self, grad, cur_round):
        self.sketch.zero()
        grad = grad.to(self.device)
        self.sketch.accumulateVec(grad[self.sketch_mask])
        if self.p2 > 0:
            server_top_k = self.sketch.unSketch(k=self.p2*self.k)
            server_hh_coords = server_top_k.nonzero()
            hhs = grad[server_hh_coords]
            server_top_k[server_hh_coords] = hhs
            weights = self._topk(server_top_k, k=self.k)
        else:
            weights = self.sketch.unSketch(k=self.k)
        weight_update = torch.zeros(self.grad_size, device=self.device)
        weight_update[self.sketch_mask] = weights
        weight_update[~self.sketch_mask] = grad[~self.sketch_mask]
        self._apply_update(weight_update, cur_round)

    def _apply_update(self, update, cur_round):
        self.cur_round = cur_round
        update = update.to(self.device)
        self.u[update.nonzero()] = 0
        self.v[update.nonzero()] = 0
        self.v[~self.sketch_mask] = 0
        start = 0
        for param_group in self.param_groups:
            for p in param_group['params']:
                end = start + torch.numel(p)
                p.data.add_(update[start:end].reshape(p.data.shape))
                start = end

    def d_star_update(self, desired_and_d, cur_round):
        desired_diff, d = desired_and_d
        self._zero_grad()
        ins = d.to(self.device)
        target = torch.zeros(1).long().to(self.device)
        out = self.model(ins)
        loss = self.criterion(out, target)
        loss.backward()
        weight_update = self._getGradVec()
        self._apply_update(weight_update, cur_round)

    def all_reduce_sketched(self, *grads):
        self.sketch.zero()
        for grad in grads:
            self.sketch.accumulateVec(grad[self.sketch_mask])
        candidate_top_k = self.sketch.unSketch(k=self.p2*self.k)
        candidate_hh_coords = candidate_top_k.nonzero()
        hhs = [grad[candidate_hh_coords] for grad in grads]
        candidate_top_k[candidate_hh_coords] = sum(hhs)
        weights = self._topk(candidate_top_k, k=self.k)
        weight_update = torch.zeros(self.grad_size, device=self.device)
        weight_update[self.sketch_mask] = weights
        weight_update[~self.sketch_mask] = sum(
                [grad[~self.sketch_mask] for grad in grads])
        self._apply_update(weight_update)

    def cpu_(self):
        self.model = self.model.cpu()
        self.u = self.u.cpu()
        self.v = self.v.cpu()
        self.sketch.cpu_()
        self.sketch_mask = self.sketch_mask.cpu()

    def cuda_(self):
        self.model = self.model.cuda()
        self.u = self.u.cuda()
        self.v = self.v.cuda()
        self.sketch.cuda_()
        self.sketch_mask = self.sketch_mask.cuda()

    def _topk(self, vec, k):
        """ Return the largest k elements (by magnitude) of vec """
        ret = torch.zeros_like(vec)
        # on a gpu, sorting is faster than pytorch's topk method
        topkIndices = torch.sort(vec**2)[1][-k:]
        ret[topkIndices] = vec[topkIndices]
        return ret

    def _getLRVec(self):
        """ Return a vector of each gradient element's learning rate
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
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.data.zero_()

    def sync(self, update):
        """Set params"""
        self._apply_update(update)
        return
        self.u[update.nonzero()] = 0
        self.v[update.nonzero()] = 0
        self.v[~self.sketch_mask] = 0
        updated_weights = update.to(self.device)
        start = 0
        for param_group in self.param_groups:
            for p in param_group['params']:
                end = start + torch.numel(p)
                p.data = updated_weights[start:end].reshape(p.data.shape)
                start = end
