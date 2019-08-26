from collections import OrderedDict 
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from csvec import CSVec

GPUS_ALLOCATED = 0.5

class FedCommEffModel:
    def __init__(self, input_model, model_cls, model_config, params):
        global client_states
        global client_params
        global param_server_states
        global cur_round
        global grad_size
        n_clients = params['n_clients']
        device = params['device']
        cpu = "cpu"
        self.model_cls = model_cls
        self.model_config = model_config
        self.cpu_model = input_model.to(cpu)
        self.model = input_model.to(device)
        param_vec = get_param_vec(self.cpu_model, cpu)
        param_vec_id = ray.put(param_vec)
        cur_round = 0
        client_params = \
                {i:
                        (cur_round, param_vec_id)
                        for i in range(n_clients)}
        param_server_states = [param_vec_id]
        grad_size = 0
        for p in self.model.parameters():
            if p.requires_grad:
                grad_size += torch.numel(p)
        if params['sketch'] or params['sketch_down']:
            global sketch
            sketch = CSVec(d=grad_size, c=params['num_cols'],
                r=params['num_rows'], device=device,
                numBlocks=1)
        self.params = params

    def train(self, training):
        self.training = training

    def __call__(self, batches, indices):
        global param_server_states
        #global client_states
        global cur_round
        global criterion
        global client_params
        if self.training:
            #batches = [(x.cpu(), y.cpu()) for x,y in batches]
            # update client state dicts
            # update rounds last updated
            if cur_round > 0:
                # update selected clients from param server
                updated_params = {
                        idx: (cur_round, client_update.remote(
                            *client_params[idx], param_server_states, cur_round,
                            self.params))
                        for idx in indices}
                client_params.update(updated_params)
            # forward pass
            grads = [forward_backward.remote(self.model_cls, self.model_config,
                *client_params[idx], *batches[i], criterion, self.params) 
                for i, idx in enumerate(indices)]
            return grads
        else:
            # param server does validation
            #batches = [batches[0].cpu(), batches[1].cpu()]
            outs = forward.remote(self.model_cls, self.model_config,
                    param_server_states[-1], *batches, self.params)
            return outs 

    def __setattr__(self, name, value):
        if name in ["training", "params", "model", "sketch", "cpu_model", "model_cls", "model_config"]:
            self.__dict__[name] = value
        else:
            self.model.setattr(name, value)

    def __getattr__(self, name):
        if name == "parameters":
            global param_server_states
            #global client_states
            #curr_state = vec_to_state_dict(ray.get(param_server_states[-1]), self.model.state_dict(), gpu)
            #self.model.load_state_dict(curr_state)
            curr_weights = ray.get(param_server_states[-1])
            set_param_vec(self.cpu_model, curr_weights)
            return getattr(self.cpu_model, name)

class FedCommEffOptimizer(optim.Optimizer):
    def __init__(self, optimizer, params):
        # extract all params from the optimizer
        self.params = params
        self.param_groups = optimizer.param_groups

    def step(self, grads, indices):
        #global client_states
        global client_params
        global param_server_states
        global cur_round
        global sketch
        lr = get_lr(self.param_groups)
        new_state = server_update.remote(indices, 
                param_server_states, self.params, lr, *grads)
        param_server_states.append(new_state)
        cur_round += 1

    def zero_grad(self):
        pass

class FedCommEffLoss:
    def __init__(self, input_criterion, params):
        global criterion
        criterion = input_criterion

@ray.remote(num_gpus=1.0)
def forward(model_cls, model_config, weights, ins, targets, params):
    device = params['device']
    model = model_cls(**model_config).to(device)
    weights = weights.to(device)
    ins = ins.to(device)
    set_param_vec(model, weights)
    outs = model(ins)
    return outs.cpu()

@ray.remote(num_gpus=1.0)
def forward_backward(model_cls, model_config, _, weights, ins, 
        targets, criterion, params):
    device = params['device']
    model = model_cls(**model_config).to(device)
    weights = weights.to(device)
    set_param_vec(model, weights)
    ins = ins.to(device)
    targets = targets.to(device)
    criterion = criterion.to(device)
    return compute_grad(model, ins, targets, criterion, params)

@ray.remote(num_gpus=GPUS_ALLOCATED)
def server_update(indices, param_server_states, 
        params, lr, *grads):
    sketched = params['sketch']
    device = torch.device(params['device'])
    grads = [grad.to(device) for grad in grads]
    curr_weights = ray.get(param_server_states[-1]).to(device)
    if sketched:
        p2 = params['p2']
        k = params['k']
        global grad_size
        sketch = CSVec(d=grad_size, c=params['num_cols'],
            r=params['num_rows'], device=device,
            numBlocks=1)
        sketch.zero()
        for grad in grads:
            sketch.accumulateVec(grad)
        if p2 > 0:
            candidate_top_k = sketch.unSketch(k=p2*k)
            candidate_hh_coords = candidate_top_k.nonzero()
            hhs = [grad[candidate_hh_coords] for grad in grads]
            candidate_top_k[candidate_hh_coords] = sum(hhs)
            weights = _topk(candidate_top_k, k=k)
        else:
            weights = sketch.unSketch(k=k)
        update = weights 
    else:
        update = torch.mean(torch.stack(grads), dim=0)
    weight_update = update * lr
    updated_weights = curr_weights - weight_update
    #print(f"{updated_weights} = {curr_weights} - {update} * {lr} from {grads}")
    return updated_weights.cpu()

@ray.remote(num_gpus=1.0)
def client_update(round_last_updated, client_weights, param_server_states,
        cur_round, params):
    device = params['device']
    stale_weights = ray.get(param_server_states[round_last_updated]).to(device)
    curr_weights = ray.get(param_server_states[-1]).to(device)
    client_weights = client_weights.to(device)
    sketch_down = params['sketch_down']
    diff_vec = curr_weights - stale_weights
    if sketch_down:
        p2 = params['p2']
        k = params['k']
        global grad_size
        sketch = CSVec(d=grad_size, c=params['num_cols'],
            r=params['num_rows'], device=device,
            numBlocks=1)
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
    else:
        weight_update = diff_vec
    updated_vec = client_weights + weight_update 
    return updated_vec.cpu()

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

def state_dict_to_vec(state_dict, device):
    return torch.cat(
        [t.reshape(-1) for t in state_dict.values()]
        ).to(device)

def vec_to_state_dict(vec, state_dict, device):
    vec = vec.to(device)
    od = OrderedDict()
    start = 0
    for key, val in state_dict.items():
        num = val.numel()
        end = start + num
        od[key] = vec[start:end].view(val.size())
        start = end
    return od

def compute_grad(model, ins, targets, criterion, params):
    outs = model(ins)
    loss = criterion(outs, targets)
    loss.backward()
    grad_vec = get_grad_vec(model)
    return grad_vec.cpu()

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

@ray.remote(num_gpus=1.0)
def update_client(round_last_updated, client_state, param_server_states,
        cur_round, params): 
    #import pdb; pdb.set_trace()
    device = torch.device(params['device'])
    client_weights = state_dict_to_vec(client_state, device)
    #print(f"{round_last_updated} < {cur_round} < {len(param_server_states)}")
    stale_weights = ray.get(param_server_states[round_last_updated]).to(device)
    curr_weights = ray.get(param_server_states[-1]).to(device)
    sketch_down = params['sketch_down']
    diff_vec = curr_weights - stale_weights
    if sketch_down:
        p2 = params['p2']
        k = params['k']
        global grad_size
        sketch = CSVec(d=grad_size, c=params['num_cols'],
            r=params['num_rows'], device=device,
            numBlocks=1)
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
    else:
        weight_update = diff_vec
    updated_vec = client_weights + weight_update 
    #print(f"{updated_vec} = {client_weights} + {weight_update}")
    updated_state = vec_to_state_dict(updated_vec, client_state)
    return updated_state.cpu()

@ray.remote(num_gpus=1.0)
def fwd_backward(model, _, state_dict, ins, targets, criterion, params):
    model.load_state_dict(state_dict)
    return compute_grad(model, ins, targets, criterion, params)

@ray.remote(num_gpus=1.0)
def fwd(model_cls, model_config, weights, base_dict, ins, targets, params):
    device = torch.device(params['device'])
    #ins = ins.to(device)
    state_dict = vec_to_state_dict(weights, base_dict)
    model = model_cls(**model_config).to(device)
    model.load_state_dict(state_dict)
    #model = model.cuda()
    out = model(ins)
    return out.cpu()

if __name__ == "__main__":
    ray.init(redis_password='functional')
    D_in, D_out, H_sizes = 2, 4, [2,4]
    n_clients = 1
    epochs, batch_size = 4, 1
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
    params = {
        'n_clients': n_clients,
        'p2': 1,
        'k': 1,
        'sketch_down': False,
        'sketch': True,
        'num_cols': 1,
        'num_rows': 1,
        #'device': 'cpu',
        'device': 'cuda',
    }
    model = FCNet(**model_config)
    model_cls = FCNet
    optimizer = optim.SGD(model.parameters(), lr=1)
    xs = torch.randn(batch_size, D_in)
    ys = torch.randn(batch_size, D_out)
    batch = [xs, ys]
    batches = [batch for _ in range(n_clients)]
    idx = [i for i in range(n_clients)]
    comm_model = FedCommEffModel(model, model_cls, model_config, params)
    comm_optimizer = FedCommEffOptimizer(optimizer, params)
    criterion = nn.MSELoss()
    comm_criterion = FedCommEffLoss(criterion, params)
    scheduler = optim.lr_scheduler.LambdaLR(comm_optimizer, 
            lambda x: x)
    for _ in range(epochs):
        comm_model.train(True)
        grads = comm_model(batches, idx)
        comm_optimizer.step(grads, idx)
        scheduler.step()
        comm_model.train(False)
        outs = comm_model(batch, idx)
        print(ray.get(outs))
