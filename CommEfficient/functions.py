import ray
import torch
import numpy as np
from csvec import CSVec
import copy
from minimal_failure import ray_free, ray_get_and_free
GPUS_ALLOCATED = 1.0
CPUS_ALLOCATED = 1
WEIGHT_ID = 0
ERROR_ID = 1
MOMENTUM_ID = 2

class FedCommEffModel:
    def __init__(self, model_cls, model_config, params):
        global client_params
        global cur_state
        global grad_size
        n_clients = params['n_clients']
        device = params['device']
        cpu = "cpu"
        self.model_cls = model_cls
        self.model_config = model_config
        input_model = model_cls(**model_config)
        if params.get('unit_test', False):
            list(input_model.parameters())[0].data.zero_()
        self.model = input_model.to(device)
        param_vec = get_param_vec(self.model, cpu)
        #param_vec_id = ray.put(param_vec)
        cur_state = param_vec
        grad_size = 0
        for p in self.model.parameters():
            if p.requires_grad:
                grad_size += torch.numel(p)
        error_vec = torch.zeros(grad_size)
        #error_vec_id = ray.put(error_vec)
        #client_params = {i: [param_vec_id, error_vec_id]
        #client_params = {i: [param_vec, error_vec]
        client_params = {i: [param_vec, error_vec, error_vec]
                         for i in range(n_clients)}
        """
        sketch_copy = CSVec(d=grad_size, c=params['num_cols'],
            r=params['num_rows'], device=device,
            numBlocks=params['num_blocks'])
        client_params = {i: [param_vec, copy.deepcopy(sketch_copy), copy.deepcopy(sketch_copy)]
                         for i in range(n_clients)}
        """
        self.params = params
        self.params['grad_size'] = grad_size

    def train(self, training):
        self.training = training

    def __call__(self, batches, indices):
        global cur_state
        global criterion
        global accuracy
        global client_params
        if self.training:
            batches = [(x.cpu(), y.cpu()) for x,y in batches]
            # update client state dicts
            # update rounds last updated
            x = ray.put(accuracy)
            outs, loss, acc, grads, weights = list(zip(
                *[update_forward_grad.remote(
                    get_weights(client_params, idx), 
                    get_error(client_params, idx),
                    cur_state, 
                    self.model_cls, 
                    self.model_config, 
                    *batches[i], 
                    criterion, 
                    x, 
                    self.params
                ) for i, idx in enumerate(indices)]))

            outs, loss, acc, grads, weights = list(outs), list(loss), list(acc), list(grads), list(weights)
            client_params = update_params(client_params, indices, weights, WEIGHT_ID)
            return outs, loss, acc, grads
        else:
            # param server does validation
            batches = [batches[0].cpu(), batches[1].cpu()]
            outs, loss, acc = forward.remote(
                    self.model_cls, self.model_config, 
                    cur_state,
                    *batches, criterion, accuracy, self.params)
            return outs, loss, acc

    def __getattr__(self, name):
        if name == "parameters":
            global cur_state
            try:
                cur_state = ray.get(cur_state)
            except:
                pass
            curr_weights = cur_state.to(self.params['device'])
            set_param_vec(self.model, curr_weights)
            return getattr(self.model, name)

class FedCommEffOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, params):
        global grad_size
        self.params = params
        self.param_groups = optimizer.param_groups
        self.server_momentum = None
        device = params['device']
        self.sketch = CSVec(d=grad_size, c=params['num_cols'],
                r=params['num_rows'], device=device,
                numBlocks=params['num_blocks'])
        momentum_copy = None
        n_momentums = 1
        if params['virtual_momentum']:
            momentum_copy = torch.zeros(grad_size)
            n_momentums = params['n_clients_per_round']
        elif params['local_momentum']:
            momentum_copy = torch.zeros(grad_size)
            n_momentums = params['n_clients']
        elif params['momentum_sketch']:
            momentum_copy = self.sketch
            #server_momentum = CSVec(d=grad_size, c=params['num_cols'],
            #    r=params['num_rows'], device=device,
            #    numBlocks=params['num_blocks'])
        self.momentums = [copy.deepcopy(momentum_copy) for _ in range(
            n_momentums)]

    def step(self, grads, indices):
        global client_params
        global cur_state
        lr = get_lr(self.param_groups)
        momentums = None
        if self.params['virtual_momentum']:
            momentums = self.momentums
        elif self.params['local_momentum']:
            momentums = [self.momentums[idx] for idx in indices]
        new_state, new_momentums, new_errors = server_update(
                indices,
                #param_server_states[-1], 
                cur_state,
                self.params, lr, 
                grads,
                momentums, 
                self.sketch)
        #new_errors = ray.get(new_errors)
        client_params = update_params(client_params, indices, new_errors, ERROR_ID)
        del cur_state
        cur_state = new_state
        if self.params['virtual_momentum']:
            self.momentums = new_momentums
        elif self.params['local_momentum']:
            for i, idx in enumerate(indices):
                self.momentums[idx] = new_momentums[i]
        """
        errors = get_params(indices, ERROR_ID)
        momentums = get_params(indices, MOMENTUM_ID)
        new_state, new_momentums, new_errors = server_update_sketched(
                cur_state,
                grads,
                momentums,
                errors,
                self.params,
                lr, self.sketch)
        del cur_state
        cur_state = new_state
        client_params = update_params(client_params, indices, new_momentums, MOMENTUM_ID)
        client_params = update_params(client_params, indices, new_errors, ERROR_ID)
        """

    def zero_grad(self):
        pass
class FedCommEffCriterion:
    def __init__(self, input_criterion, params):
        global criterion
        criterion = input_criterion
    def __call__(self, *args):
        global criterion
        out = criterion(*args)
        return out
class FedCommEffAccuracy:
    def __init__(self, input_accuracy, params):
        global accuracy
        accuracy = input_accuracy
    def __call__(self, *args):
        global accuracy
        out = accuracy(*args)
        return out

@ray.remote(num_gpus=GPUS_ALLOCATED, num_return_vals=5, num_cpus=CPUS_ALLOCATED)
def update_forward_grad(client_weights, client_error, curr_weights, model_cls,
        model_config, ins, targets, criterion, accuracy, params):
    #client_weights = ray_get_and_free(client_weights)[0]
    #client_error = ray_get_and_free(client_error)[0]
    new_client_weights = client_update(client_weights, curr_weights, params)
    #new_client_weights = curr_weights
    """
    outs, loss, acc, grad = forward_grad_sketched(model_cls, model_config,
            new_client_weights, client_error, ins, targets, criterion, 
            accuracy, params)
    """
    outs, loss, acc, grad = forward_grad(model_cls, model_config,
            new_client_weights, client_error, ins, targets, criterion, 
            accuracy, params)
    return outs, loss, acc, grad, new_client_weights.cpu()

def client_update(client_weights, curr_weights, params):
    device = params['device']
    #stale_weights = ray.get(param_server_states[round_last_updated]).to(device)
    #curr_weights = ray.get(param_server_states[-1]).to(device)
    #stale_weights = stale_weights.to(device)
    curr_weights = curr_weights.to(device)
    client_weights = client_weights.to(device)
    diff_vec = curr_weights - client_weights
    diff_vec = diff_vec.float()
    grad_size = params['grad_size']
    topk_down = params['topk_down']
    sketch_down = params['sketch_down']

    if sketch_down:
        p2 = params['p2']
        k = params['k']
        sketch = CSVec(d=grad_size, c=params['num_cols'],
            r=params['num_rows'], device=device,
            numBlocks=20)
        sketch.zero()
        sketch.accumulateVec(diff_vec)
        if p2 > 0:
            server_top_k = sketch.unSketch(k=p2*k)
            server_hh_coords = server_top_k.nonzero()
            hhs = diff_vec[server_hh_coords]
            server_top_k[server_hh_coords] = hhs
            weights = _topk(server_top_k, k=k)
        else:
            weights = sketch.unSketch(k=k)
        weight_update = weights
    elif topk_down:
        k = params['k']
        weight_update = _topk(diff_vec, k=k)
    else:
        weight_update = diff_vec
    updated_vec = client_weights + weight_update 
    #print(f"{torch.norm(weight_update, 2)}")
    #print(f"{updated_vec} = {client_weights} + {weight_update}")
    return updated_vec
    #return updated_vec.cpu()

def forward_grad(model_cls, model_config, weights, error, ins, targets,
        criterion, accuracy, params):
    device = params['device']
    model = model_cls(**model_config).to(device)
    weights = weights.to(device)
    set_param_vec(model, weights)
    ins = ins.to(device)
    targets = targets.to(device)
    criterion = criterion.to(device)
    accuracy = accuracy.to(device)
    outs = model(ins)
    loss, acc = compute_loss(outs, targets, criterion, accuracy, train=True)
    grad = get_grad(model, weights, params, train=True)
    if params['error_accum'] == 'True':
        #print(f"{error + grad} = {error} + {grad}")
        error += grad
    else:
        error = grad
    return outs.cpu(), loss, acc, error

def forward_grad_sketched(model_cls, model_config, weights, error, ins, targets,
        criterion, accuracy, params):
    device = params['device']
    model = model_cls(**model_config).to(device)
    weights = weights.to(device)
    set_param_vec(model, weights)
    ins = ins.to(device)
    targets = targets.to(device)
    criterion = criterion.to(device)
    accuracy = accuracy.to(device)
    outs = model(ins)
    loss, acc = compute_loss(outs, targets, criterion, accuracy, train=True)
    grad = get_grad(model, weights, params, train=True, device=device)
    sketch = CSVec(d=params['grad_size'], c=params['num_cols'],
        r=params['num_rows'], device=device,
        numBlocks=params['num_blocks'])
    sketch.accumulateVec(grad)
    return outs.cpu(), loss, acc, sketch.table.cpu()

@ray.remote(num_gpus=GPUS_ALLOCATED, num_return_vals=3)
def forward(model_cls, model_config, weights, ins, targets,
        criterion, accuracy, params):
    device = params['device']
    model = model_cls(**model_config).to(device)
    weights = weights.to(device)
    set_param_vec(model, weights)
    ins = ins.to(device)
    targets = targets.to(device)
    criterion = criterion.to(device)
    accuracy = accuracy.to(device)
    outs = model(ins)
    loss, acc = compute_loss(outs, targets, criterion, accuracy, train=False)
    return outs.cpu(), loss, acc

def server_update_sketched(curr_weights, grads, momentums, errors, params, lr, sketch):
    """
    We have W sketches for momentum and error; S(u_i), S(v_i)
    Workers send us S(g_i) and we do:
    S(u_i) = 0.9 * S(u_i) + S(g_i)
    S(v_i) += S(u_i)
    weight_update = sum(S(v_i)).unSketch()
    S(v_i).accumulateVec(-1*weightUpdate)
    S(u_i).accumulateVec(-1*weightUpdate)
    """
    momentum = params['momentum']
    k = params['k']
    device = torch.device(params['device'])
    curr_weights = curr_weights.to(device)
    grads = [ray.get(grad).to(device) for grad in grads]
    #print(f"{errors[0].table} = {momentum} * {momentums[0].table} + {grads[0]}")
    for grad, u, v in zip(grads, momentums, errors):
        if params['momentum_sketch']:
            u *= momentum
            #print(f"Accumulating {grad} into {u.table}")
            u.accumulateTable(grad)
            #print(f"Adding {u.table} to {v.table}")
            v += u
        else:
            v.accumulateTable(grad)
    #print(f"{errors[0].table} = {momentum} * {momentums[0].table} + {grads[0]}")
    update = np.sum(errors).unSketch(k=k)
    neg_update = -1. * update 
    sketch.zero()
    sketch.accumulateVec(neg_update)
    hh_coords = sketch.table.nonzero()
    import pdb; pdb.set_trace()
    for u, v in zip(momentums, errors):
        if params['momentum_sketch']:
            u.table[hh_coords] = 0
        v.table[hh_coords] = 0
    """
    for u, v in zip(momentums, errors):
        if params['momentum_sketch']:
            u.accumulateVec(neg_update)
        v.accumulateVec(neg_update)
    """
    weight_update = update * lr
    updated_weights = curr_weights - weight_update
    #print(f"{updated_weights} = {curr_weights} - {weight_update} ({update} * {lr}) from {grads}")
    return updated_weights.cpu(), momentums, errors

#@ray.remote(num_gpus=GPUS_ALLOCATED, num_return_vals=3)
def server_update(indices, curr_weights,
        params, lr, grads, momentums, sketch):
    device = torch.device(params['device'])
    #try:
    #grads = ray_get_and_free(grads)
    grads = [ray.get(grad).to(device) for grad in grads]
   # except:
    #grads = [grad.to(device) for grad in grads]
    #momentums = [ray.get(momentum).to(device) for momentum in momentums]
    #curr_weights = ray.get(param_server_states[-1]).to(device)
    #curr_weights = ray.get(curr_weights)
    curr_weights = curr_weights.to(device)
    k = params['k']

    if params['sketch']:
        p2 = params['p2']
        sketch.zero()
        if params['momentum_sketch']:
            for g, u in zip(grads, momentums):
                u *= params['momentum']
                u += g
        if params['local_momentum']:
            momentums = [u.to(device) for u in momentums]
            for g, u in zip(grads, momentums):
                u.mul_(params['momentum'])
                u.add_(1,g)
        if params["grad_reduce"] == "median":
            sketches = [copy.deepcopy(sketch) for _ in grads]
            if params['local_momentum']:
                for i, u in enumerate(momentums):
                    sketches[i].accumulateVec(u)
            else:
                for i, grad in enumerate(grads):
                    sketches[i].accumulateVec(grad)
            sketch = CSVec.median(sketches)
        else:
            for grad in grads:
                sketch.accumulateVec(grad)
            if params["grad_reduce"] == "mean":
                sketch /= len(grads)
        if p2 > 0:
            candidate_top_k = sketch.unSketch(k=p2*k)
            candidate_hh_coords = candidate_top_k.nonzero()
            if params['local_momentum']:
                hhs = [grad[candidate_hh_coords] for grad in momentums]
            else:
                hhs = [grad[candidate_hh_coords] for grad in grads]
            candidate_top_k[candidate_hh_coords] = sum(hhs)
            update = _topk(candidate_top_k, k=k)
        else:
            update = sketch.unSketch(k=k)
        if params['virtual_momentum']:
            u = momentums[0].to(device)
            u.mul_(params['momentum'])
            u.add_(1, update)
            momentums = [u.cpu()]

    elif params['true_topk']:
        update = _topk(torch.sum(torch.stack(grads), dim=0), k=k)
    elif params['local_topk']:
        local_topks = [_topk(grad, k=k) for grad in grads]
        update = _topk(torch.sum(torch.stack(local_topks), dim=0), k=k)
    else:
        if params['virtual_momentum'] or params['local_momentum']:
            momentums = [u.to(device) for u in momentums]
            for g, u in zip(grads, momentums):
                u.mul_(params['momentum'])
                u.add_(1,g)
            update = torch.sum(torch.stack(momentums), dim=0)
            #grads = [u.mul_(params['momentum']).add_(
            #    1, g) for g, u in zip(grads, momentum)]
            momentums = [u.cpu() for u in momentums]
        else:
            update = torch.sum(torch.stack(grads), dim=0)
    weight_update = update * lr
    updated_weights = curr_weights - weight_update
    #print(f"{updated_weights} = {curr_weights} - {weight_update} from {grads}")
    for grad in grads:
        grad[weight_update.nonzero()] = 0
    #grads = [ray.put(u.cpu()) for u in grads]
    grads = [u.cpu() for u in grads]
    if params['local_momentum']:
        momentums = [u.cpu() for u in momentums]
    return updated_weights.cpu(), momentums, grads

def get_weights(client_params, idx):
    retval = client_params[idx][WEIGHT_ID]
    return retval

def get_error(client_params, idx):
    retval = client_params[idx][ERROR_ID]
    return retval

def get_errors(client_params, indices):
    return [client_params[idx][ERROR_ID] for idx in indices]

def get_params(indices, idx):
    return [client_params[index][idx] for index in indices]

def update_params(client_params, client_indices, vecs, vec_idx):
    for i, idx in enumerate(client_indices):
        #ray_free(client_params[idx][vec_idx])
        try:
            client_params[idx][vec_idx] = ray.get(vecs[i])
        except:
            client_params[idx][vec_idx] = vecs[i]
    return client_params

def get_lr(optimizer_param_groups):
    if len(optimizer_param_groups) == 1:
        lr = optimizer_param_groups[0]["lr"]
        #print(f"Lr is {lr}")
        return lr

def _topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    ret = torch.zeros_like(vec)
    # on a gpu, sorting is faster than pytorch's topk method
    topkIndices = torch.sort(vec**2)[1][-k:]
    ret[topkIndices] = vec[topkIndices]
    return ret

def get_grad(model, weights, params, train, device='cpu'):
    if train:
        grad_vec = get_grad_vec(model)
        if params['weight_decay'] != 0:
            grad_vec.add_(params['weight_decay']/params['n_clients_per_round'],
                weights)
        return grad_vec.to(device)
    else:
        return 0

def compute_loss(outs, targets, criterion, accuracy, train):
    loss = criterion(outs, targets)
    if train:
        loss.backward()
    batch_loss = loss.mean().cpu().detach().numpy()
    acc = accuracy(outs, targets).float().mean().cpu().detach().numpy()
    return batch_loss, acc

def get_grad_vec(model):
    grad_vec = []
    with torch.no_grad():
        # flatten
        for p in model.parameters():
            if p.grad is None:
                grad_vec.append(torch.zeros_like(p.data.view(-1)))
            else:
                grad_vec.append(p.grad.data.view(-1).float())
        # concat into a single vector
        grad_vec = torch.cat(grad_vec)
    return grad_vec

def get_param_vec(model, device):
    return torch.cat([p.data.view(-1) for p in model.parameters()]).to(device)
    param_vec = []
    for p in model.parameters():
        param_vec.append(p.data.view(-1))
    return torch.cat(param_vec).to(device)

def set_param_vec(model, param_vec):
    start = 0
    for p in model.parameters():
        end = start + p.numel()
        p.data.zero_()
        p.data.add_(param_vec[start:end].view(p.size()))
        start = end

@ray.remote
def identity(x):
    return x
