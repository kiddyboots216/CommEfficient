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
    def __init__(self, input_model, params):
        global client_states
        global param_server_states
        global cur_round
        global grad_size
        n_clients = params['n_clients']
        device = params['device']
        cpu = torch.device("cpu")
        self.model = input_model.to(device)
        state_dict = input_model.state_dict()
        param_vec = get_param_vec(input_model, cpu)
        state_dict_id = ray.put(state_dict)
        param_vec_id = ray.put(param_vec)
        cur_round = 0
        client_states = \
            {i:
                (cur_round, state_dict_id)
            for i in range(n_clients)}
        client_params = \
                {i:
                        (cur_round, param_vec_id)
                        for i in range(n_clients)}
        #client_states = {0: (cur_round, state_dict_id)}
        param_server_states = [param_vec_id]
        self.grad_size = 0
        for p in self.model.parameters():
            if p.requires_grad:
                self.grad_size += torch.numel(p)
        if params['sketch'] or params['sketch_down']:
            self.sketch = CSVec(d=self.grad_size, c=params['num_cols'],
                r=params['num_rows'], device=cpu,
                numBlocks=1)
        self.params = params

    def train(self, training):
        self.training = training

    def __call__(self, batches, indices):
        global param_server_states
        global client_states
        global cur_round
        global optimizer_param_groups
        global criterion
        if self.training:
            #batches = [(x.cpu(), y.cpu()) for x,y in batches]
            #batches = [batches[0].cpu(), batches[1].cpu()]
            # update client state dicts
            # update rounds last updated
            if cur_round > 0:
                # update selected clients from param server
                updated_states = {
                    idx: (cur_round, update_client.remote(
                        *client_states[idx], param_server_states, cur_round,
                        self.params, optimizer_param_groups, self.sketch
                        )) 
                    for idx in indices}
                client_states.update(updated_states)
            # forward pass
            grads = [fwd_backward.remote(self.model, *client_states[idx], 
                *batches[i], criterion) for i, idx in enumerate(indices)]
            return grads
        else:
            # param server does validation
            #batches = [batches[0].cpu(), batches[1].cpu()]
            outs = forward.remote(self.model, param_server_states[-1], 
                    client_states[0][-1], *batches, self.params)
            return outs 

    def __setattr__(self, name, value):
        if name in ["training", "params", "model", "sketch"]:
            self.__dict__[name] = value
        else:
            self.model.setattr(name, value)

    def __getattr__(self, name):
        if name == "parameters":
            global param_server_states
            global client_states
            gpu = torch.device(self.params['gpu'])
            curr_state = vec_to_state_dict(ray.get(param_server_states[-1]), self.model.state_dict(), gpu)
            self.model.load_state_dict(curr_state)
            return getattr(self.model, name)

class FedCommEffOptimizer(optim.Optimizer):
    def __init__(self, optimizer, params):
        global optimizer_param_groups
        # extract all params from the optimizer
        self.params = params
        optimizer_params = [
                {'lr': p['lr'],
                'dampening': p['dampening'],
                'nesterov': p['nesterov'],
                'momentum': p['momentum'],
                'weight_decay': p['weight_decay']
                } for p in optimizer.param_groups
        ]
        self.param_groups = optimizer.param_groups
        optimizer_param_groups = self.param_groups
        if params['sketch'] or params['sketch_down']:
            self.sketch = CSVec(d=grad_size, c=params['num_cols'],
                r=params['num_rows'], device=params['device'],
                numBlocks=1)

    def step(self, grads, indices):
        global client_states
        global param_server_states
        global cur_round
        new_state = server_update.remote(grads, indices, client_states, 
                param_server_states, self.params, self.sketch, 
                self.param_groups)
        param_server_states.append(new_state)
        cur_round += 1

    def zero_grad(self):
        pass

class FedCommEffLoss:
    def __init__(self, input_criterion, params):
        global criterion
        criterion = input_criterion

@ray.remote(num_gpus=GPUS_ALLOCATED)
def server_update(grads, indices, client_states, param_server_states, params, sketch, optimizer_param_groups):
    sketched = params['sketch']
    device = torch.device(params['device'])
    grads = [ray.get(grad).to(device) for grad in grads]
    lr = get_lr(optimizer_param_groups)
    if sketched:
        p2 = params['p2']
        k = params['k']
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
        update = torch.mean(torch.stack(grads), dim=0), ray.get(client_states[0][-1])
    curr_weights = ray.get(param_server_states[-1])
    weight_update = update * lr
    updated_weights = curr_weights - weight_update
   # print(f"{updated_weights} = {curr_weights} - {weight_update}")
    return updated_weights

@ray.remote(num_gpus=1.0)
def update_client(round_last_updated, client_state, param_server_states,
        cur_round, params, optimizer_param_groups, sketch
        ): 
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
        sketch.cuda_()
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

def get_lr(optimizer_param_groups):
    if len(optimizer_param_groups) == 1:
        lr = optimizer_param_groups[0]["lr"]
        print(f"Lr is {lr}")
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

@ray.remote(num_gpus=1.0)
def forward(model, weights, base_dict, ins, targets, params):
    device = torch.device(params['device'])
    #ins = ins.to(device)
    state_dict = vec_to_state_dict(weights, base_dict)
    model.load_state_dict(state_dict)
    #model = model.cuda()
    out = model(ins)
    return out

@ray.remote(num_gpus=1.0)
def fwd_backward(model, _, state_dict, ins, targets, criterion):
    model.load_state_dict(state_dict)
    outs = model(ins)
    loss = criterion(outs, targets)
    loss.backward()
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

if __name__ == "__main__":
    ray.init(redis_password='functional')
    D_in, D_out, H_sizes = 2, 4, [2,4]
    n_clients = 1
    epochs, batch_size = 10, 1
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
    optimizer = optim.SGD(model.parameters(), lr=1)
    xs = torch.randn(batch_size, D_in)
    ys = torch.randn(batch_size, D_out)
    batch = [xs, ys]
    batches = [batch for _ in range(n_clients)]
    idx = [i for i in range(n_clients)]
    comm_model = FedCommEffModel(model, params)
    optimizer = FedCommEffOptimizer(optimizer, params)
    criterion = nn.MSELoss().to(device)
    comm_criterion = FedCommEffLoss(criterion, params)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
            lambda x: x)
    for _ in range(epochs):
        comm_model.train(True)
        grads = comm_model(batches, idx)
        optimizer.step(grads, idx)
        scheduler.step()
        comm_model.train(False)
        outs = comm_model(batch, idx)
        print(ray.get(outs))
