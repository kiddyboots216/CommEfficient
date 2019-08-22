from collections import OrderedDict 
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from CommEfficient.minimal import CSVec
GPUS_ALLOCATED = 0.5
class FedCommEffModel:
    def __init__(self, input_model, params):
        n_clients = params['n_clients']
        global model
        model = input_model
        state_dict = model.state_dict()
        global client_states
        client_states = \
            {i:
                (0, ray.put(state_dict))
            for i in range(n_clients)}
        global param_server_states
        param_server_states = [ray.put(state_dict)]
        global cur_round
        cur_round = 0
        self.params = params

    def train(self, training):
        self.training = training

    def __call__(self, *args):
        global param_server_states
        if self.training:
            # update client state dicts
            # update rounds last updated
            global client_states
            global cur_round
            batches, indices = args
            if cur_round > 0:
                updated_states = {}
                for idx in indices:
                    #import pdb; pdb.set_trace()
                    updated_states[idx] = (cur_round, update_client.remote(client_states[idx], self.params))
                """
                updated_states = \
                    {i:
                        (cur_round,
                        update_client.remote(client_states[i], params))
                    for i in indices}
                """
                client_states.update(updated_states)
                print(f"{cur_round} < {len(param_server_states)}")
            # forward pass
            grads = []
            for i, idx in enumerate(indices):
                grads.append(fwd_backward.remote(client_states[idx], batches[i]))
            #grads = [fwd_backward.remote(client_states[idx], 
            #    batches[i]) for i, idx in enumerate(indices)]
            return grads
        else:
            # param server does validation
            outs = forward.remote(param_server_states[-1], *args)
            return outs 
    """
    def __setattr__(self, name, value):
        if name in ["training", "params"]:
            self.__dict__[name] = value
        else:
            global model
            setattr(model, name, value)

    def __getattr__(self, name):
        global model
        return getattr(model, name)
    """

class FedCommEffOptimizer(optim.Optimizer):
    def __init__(self, optimizer, params):
        # extract all params from the optimizer
        global optimizer_param_groups
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

    def step(self, grads, indices):
        new_state = server_update.remote(grads, indices)
        global param_server_states
        param_server_states.append(new_state)
        global cur_round
        cur_round += 1
        #print(f"{cur_round} < {len(param_server_states)}")

    def zero_grad(self):
        pass

class FedCommEffLoss:
    def __init__(self, input_criterion, params):
        global criterion
        criterion = input_criterion

@ray.remote(num_gpus=GPUS_ALLOCATED)
def server_update(grads, indices):
    grads = [ray.get(grad) for grad in grads]
    global client_states
    return vec_to_state_dict(torch.mean(torch.stack(grads), dim=0), ray.get(client_states[0][-1]))

@ray.remote(num_gpus=GPUS_ALLOCATED)
def sketched_server_update(grads, indices, params):
    p2 = params['p2']
    k = params['k']
    global sketch
    sketch.zero()
    for grad in grads:
        sketch += grad
    if p2 > 0:
        candidate_top_k = sketch.unSketch(k=p2*k)
        candidate_hh_coords = candidate_top_k.nonzero()
        hhs = [grad[candidate_hh_coords] for grad in grads]
        candidate_top_k[candidate_hh_coords] = sum(hhs)
        weights = _topk(candidate_top_k, k=k)
    else:
        weights = sketch.unSketch(k=k)
    weight_update = weights
    return weight_update

@ray.remote(num_gpus=GPUS_ALLOCATED)
def update_client(round_state_tuple, params, sketch_down=False):
    #import pdb; pdb.set_trace()
    round_last_updated, client_state = round_state_tuple
    client_state = ray.get(client_state)
    device = torch.device("cuda")
    client_weights = state_dict_to_vec(client_state, device)
    global param_server_states
    global cur_round
    print(f"{round_last_updated} == {cur_round} < {len(param_server_states)}")
    stale_state = ray.get(param_server_states[round_last_updated])
    stale_weights = state_dict_to_vec(stale_state, device)
    curr_state = ray.get(param_server_states[-1])
    curr_weights = state_dict_to_vec(curr_state, device)
    if sketch_down:
        p2 = params['p2']
        k = params['k']
        global sketch
        sketch.zero()
        sketch += grad
        if p2 > 0:
            server_top_k = sketch.unSketch(k=p2*k)
            server_hh_coords = server_top_k.nonzero()
            hhs = grad[server_hh_coords]
            server_top_k[server_hh_coords] = hhs
            weights = _topk(server_top_k, k=k)
        else:
            weights = sketch.unSketch(k=k)
        weight_update = weights
    else:
        weight_update = curr_weights - stale_weights
    updated_vec = client_weights + weight_update * get_lr()
    updated_vec = updated_vec.cpu()
    updated_state = vec_to_state_dict(updated_vec, client_state)
    return updated_state

def get_lr():
    global optimizer_param_groups
    if len(optimizer_param_groups) == 1:
        lr = optimizer_param_groups[0]["lr"]
        print(f"Lr is {lr}")
        return lr

def state_dict_to_vec(state_dict, device):
    return torch.cat(
        [tensor.reshape(-1) for tensor in state_dict.values()]
        ).to(device)

def vec_to_state_dict(vec, state_dict):
    od = OrderedDict()
    start = 0
    for key, val in state_dict.items():
        num = val.numel()
        end = start + num
        od[key] = vec[start:end].view(val.size())
        start = end
    return od

@ray.remote(num_gpus=GPUS_ALLOCATED)
def forward(state_dict, *args):
    batch = args[0]
    ins, targets = batch
    device = torch.device("cuda")
    ins = ins.to(device)
    global model
    model.load_state_dict(state_dict)
    mode = model.cuda()
    out = model(ins)
    return out.cpu()

@ray.remote(num_gpus=GPUS_ALLOCATED)
def fwd_backward(round_state_tuple, batch):
    ins, targets = batch
    ins, targets = ins.cuda(), targets.cuda()
    _, state_dict = round_state_tuple
    state_dict = ray.get(state_dict)
    global model
    global criterion
    model.load_state_dict(state_dict)
    model = model.cuda()
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
    device = torch.device("cpu")
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
    model = FCNet(**model_config)
    optimizer = optim.SGD(model.parameters(), lr=1)
    params = {
        'n_clients': n_clients,
        'p2': 1,
        'k': 1,
    }
    xs = torch.randn(batch_size, D_in, device=device)
    ys = torch.randn(batch_size, D_out, device=device)
    batch = [xs, ys]
    batches = [batch for _ in range(n_clients)]
    idx = [i for i in range(n_clients)]
    comm_model = FedCommEffModel(model, params)
    optimizer = FedCommEffOptimizer(optimizer, params)
    criterion = nn.MSELoss().cuda()
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
        print(ray.get(outs).mean())
