import ray
import torch
from csvec import CSVec
GPUS_ALLOCATED = 1.0

class FedCommEffModel:
    def __init__(self, model_cls, model_config, params):
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
        input_model = model_cls(**model_config)
        if params.get('unit_test', False):
            list(input_model.parameters())[0].data.zero_()
        self.model = input_model.to(device)
        param_vec = get_param_vec(self.model, cpu)

        param_vec_id = ray.put(param_vec)
        param_server_states = [param_vec_id]
        grad_size = 0
        for p in self.model.parameters():
            if p.requires_grad:
                grad_size += torch.numel(p)
        """
        momentum_vec_id = None
        if params['client_momentum']:
            momentum_vec = torch.zeros(grad_size, device=device)
            momentum_vec_id = ray.put(momentum_vec)
        """
        cur_round = 0
        client_params = \
                {i:
                        (cur_round, param_vec_id)
                        for i in range(n_clients)}
        self.params = params

    def train(self, training):
        self.training = training

    def __call__(self, batches, indices):
        global param_server_states
        global cur_round
        global criterion
        global accuracy
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
                            self.params, server_momentum))
                        for idx in indices}
                client_params.update(updated_params)

            # forward pass
            import pdb; pdb.set_trace()
            outs, loss, acc, grads = list(zip(*[forward_grad.remote(
                self.model_cls, self.model_config, 
                get_weights(client_params, idx),
                *batches[i], criterion, accuracy, self.params, self.training)
                for i, idx in enumerate(indices)]))
            return list(outs), list(loss), list(acc), list(grads)

        else:
            # param server does validation
            #batches = [batches[0].cpu(), batches[1].cpu()]
            outs, loss, acc, _ = forward_grad.remote(
                    self.model_cls, self.model_config, 
                    param_server_states[-1], 
                    *batches, criterion, accuracy, self.params, self.training)
            return outs, loss, acc

    def __getattr__(self, name):
        if name == "parameters":
            global param_server_states
            curr_weights = ray.get(param_server_states[-1]).to(
                    self.params['device'])
            set_param_vec(self.model, curr_weights)
            return getattr(self.model, name)

class FedCommEffOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, params):
        global grad_size
        self.params = params
        self.param_groups = optimizer.param_groups
        self.server_momentum = None
        device = params['device']
        server_momentum = None
        if params.get('virtual_momentum', False):
            server_momentum = torch.zeros(grad_size)
        if params.get('momentum_sketch', False):
            server_momentum = CSVec(d=grad_size, c=params['num_cols'],
                r=params['num_rows'], device=device,
                numBlocks=params['num_blocks'])
            self.server_momentum = ray.put(server_momentum_sketch)
        vec_id = ray.put(server_momentum)
        self.server_momentums = [vec_id for _ in range(
            params['n_clients_per_round'])]

    def step(self, grads, indices):
        global client_params
        global param_server_states
        global cur_round
        #global sketch
        # select momentum
        lr = get_lr(self.param_groups)
        new_state, new_momentum = server_update.remote(indices, 
                param_server_states, self.params, lr, 
                grads, self.server_momentums)
        param_server_states.append(new_state)
        self.server_momentums = new_momentum
        cur_round += 1

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

@ray.remote(num_gpus=GPUS_ALLOCATED, num_return_vals=4)
def forward_grad(model_cls, model_config, weights, ins, targets,
        criterion, accuracy, params, train):
    device = params['device']
    model = model_cls(**model_config).to(device)
    weights = weights.to(device)
    set_param_vec(model, weights)
    ins = ins.to(device)
    targets = targets.to(device)
    criterion = criterion.to(device)
    accuracy = accuracy.to(device)
    outs = model(ins)
    loss, acc = compute_loss(outs, targets, criterion, accuracy, train=train)
    grads = get_grad(model, weights, params, train=train)
    return outs.cpu(), loss, acc, grads

@ray.remote(num_gpus=GPUS_ALLOCATED, num_return_vals=2)
def server_update(indices, param_server_states,
        params, lr, grads, momentums):
    sketched = params['sketch']
    device = torch.device(params['device'])
    grads = [ray.get(grad).to(device) for grad in grads]
    momentums = [ray.get(momentum).to(device) for momentum in momentums]
    curr_weights = ray.get(param_server_states[-1]).to(device)

    if sketched:
        p2 = params['p2']
        k = params['k']
        global grad_size
        sketch = CSVec(d=grad_size, c=params['num_cols'],
            r=params['num_rows'], device=device,
            numBlocks=1)
        sketch.zero()
        if params.get('momentum_sketch', False):
            grads = [g + params['momentum'] * u for g, u in zip(grads, momentum)]
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
        if params.get('virtual_momentum', False):
            for g, u in zip(grads, momentums):
                u.mul_(params['momentum'])
                u.add_(1,g)
            grads = momentums
            #grads = [u.mul_(params['momentum']).add_(
            #    1, g) for g, u in zip(grads, momentum)]
        update = torch.mean(torch.stack(grads), dim=0)
    weight_update = update * lr
    updated_weights = curr_weights - weight_update
    #print(f"{updated_weights} = {curr_weights} - {update} * {lr} from {grads}")
    grads = [grad.cpu() for grad in grads]
    return updated_weights.cpu(), grads

@ray.remote(num_gpus=GPUS_ALLOCATED)
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

def get_weights(client_params, idx):
    return client_params[idx][1]

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

def get_grad(model, weights, params, train):
    if train:
        grad_vec = get_grad_vec(model)
        grad_vec.add_(params['weight_decay']/params['n_clients_per_round'],
                weights)
        return grad_vec.cpu()
    else:
        return 0

def compute_loss(outs, targets, criterion, accuracy, train):
    loss = criterion(outs, targets)
    if train:
        loss.sum().backward()
    batch_loss = loss.mean().cpu().detach().numpy()
    acc = accuracy(outs, targets).float().mean().cpu().detach().numpy()
    return batch_loss, acc

def get_grad_vec(model):
    return torch.cat([torch.zeros_like(p.data.view(-1)) if p.grad is None else
                    p.data.view(-1).float() for p in model.parameters()])
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

