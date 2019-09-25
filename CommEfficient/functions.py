import torch
from datetime import datetime
import os
import logging
import numpy as np
from csvec import CSVec
import copy

import ctypes
import multiprocessing
from multiprocessing.sharedctypes import RawArray
from multiprocessing import Array
#from multiprocessing import shared_memory

GPUS_ALLOCATED = 1.0
CPUS_ALLOCATED = 1
WEIGHT_ID = 0
ERROR_ID = 1
MOMENTUM_ID = 2

#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#world_size = comm.Get_size()

g_worker_Sgrads_sm = None
g_client_weights_sm = None
g_ps_weights_sm = None

g_criterion = None
g_accuracy = None

def make_logdir(params: dict):
    rows = params["num_rows"]
    cols = params["num_cols"]
    k = params["k"]
    sketch = params["sketch"]
    sketch_str = f"{rows} x {cols} : {k}" if sketch else "False"
    workers = params["n_workers"]
    clients = params["n_clients"]
    clients_str = f"{workers}/{clients}"
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + clients_str + '_' + sketch_str)
    return logdir

def init_pool(worker_Sgrads_sm, client_weights_sm, ps_weights_sm):
    global gw_worker_Sgrads_sm, gw_client_weights_sm, gw_ps_weights_sm
    gw_worker_Sgrads_sm = worker_Sgrads_sm
    gw_client_weights_sm = client_weights_sm
    gw_ps_weights_sm = ps_weights_sm

def sm2np(sm, shape, dtype=ctypes.c_float):
    nparray = np.ndarray(shape, dtype=dtype, buffer=sm)
    #nparray = np.frombuffer(sm, dtype=ctypes.c_float)
    assert(nparray.base is sm)
    return nparray

class FedCommEffModel:
    def __init__(self, model_cls, model_config, params):
        n_clients = params['n_clients']
        participation = params["participation"]
        device = params['device']
        cpu = "cpu"
        self.model_cls = model_cls
        self.model_config = model_config
        input_model = model_cls(**model_config)
        if params.get('unit_test', False):
            list(input_model.parameters())[0].data.zero_()
        self.model = input_model.to(device)
        param_vec = get_param_vec(self.model, cpu).numpy()
        #param_vec_id = ray.put(param_vec)
        grad_size = 0
        for p in self.model.parameters():
            if p.requires_grad:
                grad_size += torch.numel(p)
        """
        error_vec = torch.zeros(grad_size)
        #error_vec_id = ray.put(error_vec)
        #client_weights = {i: [param_vec_id, error_vec_id]
        #client_weights = {i: [param_vec, error_vec]
        client_weights = {i: [param_vec, error_vec, error_vec]
                         for i in range(n_clients)}
        """
        """
        sketch_copy = CSVec(d=grad_size, c=params['num_cols'],
            r=params['num_rows'], device=device,
            numBlocks=params['num_blocks'])
        client_weights = {i: [param_vec, copy.deepcopy(sketch_copy), copy.deepcopy(sketch_copy)]
                         for i in range(n_clients)}
        """

        global g_worker_Sgrads_sm
        global g_client_weights_sm
        global g_ps_weights_sm

        # ps_weights needs to be in shared memory so the workers can
        # update themselves with (possibly an approximation of) the
        # latest PS weights

        g_ps_weights_sm = Array(ctypes.c_float,
                                param_vec.size,
                                lock=False)
        ps_weights = sm2np(g_ps_weights_sm, param_vec.shape)
        # store the initial weights of the model
        ps_weights[:] = param_vec[:]

        # client weights emulates each client's possibly stale weights
        g_client_weights_sm = Array(ctypes.c_float,
                                    param_vec.size * n_clients,
                                    lock=False)

        # copy ps_weights into every row of client_weights
        client_weights = sm2np(g_client_weights_sm,
                               (n_clients, param_vec.size))
        client_weights[:] = np.tile(param_vec, (n_clients, 1))

        # this shared memory block will hold the gradient sketches computed
        # by each worker in a round
        n_workers = params["n_workers"]
        g_worker_Sgrads_sm = Array(
                'f',
                n_workers * params["num_rows"] * params["num_cols"],
                lock=False
            )
        # and zero out worker_Sgrads to start
        worker_Sgrads_shape = (n_workers, params["num_rows"], params["num_cols"])
        worker_Sgrads = sm2np(g_worker_Sgrads_sm, worker_Sgrads_shape)
        worker_Sgrads[:] = 0

        self.process_pool = multiprocessing.Pool(
                n_workers,
                initializer=init_pool,
                initargs=(g_worker_Sgrads_sm, g_client_weights_sm,
                          g_ps_weights_sm)
            )

        self.params = params
        self.params['grad_size'] = grad_size

    def train(self, training):
        self.training = training

    def __call__(self, batches, indices):
        global g_criterion
        global g_accuracy
        if self.training:
            # batches = [(x.cpu(), y.cpu()) for x,y in batches]
            # update client state dicts
            #x = ray.put(accuracy)
            """
            outs, loss, acc, grads, weights = list(zip(
                *[update_forward_grad.remote(
                    get_weights(client_weights, idx),
                    get_error(client_weights, idx),
                    ps_weights,
                    self.model_cls,
                    self.model_config,
                    *batches[i],
                    criterion,
                    x,
                    self.params
                ) for i, idx in enumerate(indices)]))
            """
            args_tuples = [(i, idx, self.model_cls, self.model_config,
                            batches[i][0], batches[i][1], self.params,
                            g_criterion, g_accuracy)
                           for i, idx in enumerate(indices)]
            results = self.process_pool.starmap(update_forward_grad_sketched,
                                                args_tuples)
            """
            processes = []
            for i in range(len(indices)):
                p = Process(target=update_forward_grad_sketched, args=args_tuples[i])
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            """
            outs = np.array([r[0] for r in results])
            loss = np.array([r[1] for r in results])
            acc = np.array([r[2] for r in results])

            return outs, loss, acc
        else:
            # param server does validation
            # batches = [batches[0].cpu(), batches[1].cpu()]
            outs, loss, acc = forward(
                    self.model_cls, self.model_config,
                    *batches,
                    #criterion, accuracy,
                    self.params)
            return outs, loss, acc

    def __getattr__(self, name):
        if name == "parameters":
            global g_ps_weights_sm

            ps_weights = sm2np(g_ps_weights_sm,
                               (self.params["grad_size"],))
            curr_weights = torch.from_numpy(ps_weights)
            curr_weights = curr_weights.to(self.params['device'])
            set_param_vec(self.model, curr_weights)
            return getattr(self.model, name)

class FedCommEffOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, params):
        #global grad_size
        grad_size = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    grad_size += torch.numel(p)
        self.params = params
        self.param_groups = optimizer.param_groups
        device = params['device']
        self.sketch = CSVec(d=grad_size, c=params['num_cols'],
                r=params['num_rows'], device=device,
                numBlocks=params['num_blocks'])
        n_momentums = 1
        momentum_copy = None
        n_errors = 1
        error_copy = None
        if params['do_virtual_momentum_sketch']:
            n_momentums = params['n_workers']
            momentum_copy = self.sketch
        elif params['do_local_momentum_sketch']:
            n_momentums = params['n_clients']
            momentum_copy = self.sketch
        self.momentums = [copy.deepcopy(momentum_copy) for _ in range(
            n_momentums)]
        if params['do_virtual_error_sketch']:
            n_errors = params['n_workers']
            error_copy = self.sketch
        elif params['do_local_error_sketch']:
            n_errors = params['n_clients']
            error_copy = self.sketch
        self.errors = [copy.deepcopy(error_copy) for _ in range(
            n_errors)]

    def get_lr(self):
        lr = get_lr(self.param_groups)
        return lr

    def step(self, indices):
        #global client_weights
        lr = self.get_lr()
        momentums = [None for _ in indices]
        """
        if self.params['virtual_momentum']:
            momentums = self.momentums
        elif self.params['local_momentum']:
            momentums = [self.momentums[idx] for idx in indices]
        """
        if self.params['do_virtual_momentum_sketch']:
            momentums = self.momentums
        elif self.params['do_local_momentum_sketch']:
            momentums = [self.momentums[idx] for idx in indices]
        """
        new_state, new_momentums, new_errors = server_update(
                indices,
                #param_server_states[-1],
                ps_weights,
                self.params, lr,
                grads,
                momentums,
                self.sketch)
        #new_errors = ray.get(new_errors)
        client_weights = update_params(client_weights, indices, new_errors, ERROR_ID)
        del ps_weights
        ps_weights = new_state
        if self.params['virtual_momentum']:
            self.momentums = new_momentums
        elif self.params['local_momentum']:
            for i, idx in enumerate(indices):
                self.momentums[idx] = new_momentums[i]
        """
        #errors = get_params(indices, ERROR_ID)
        errors = self.errors
        if self.params['do_virtual_error_sketch']:
            errors = self.errors
        elif self.params['do_local_error_sketch']:
            errors = [self.errors[idx] for idx in indices]
        new_ps_weights, new_momentums, new_errors = get_updated_server_sketched(
                momentums,
                errors,
                self.params,
                lr, self.sketch)

        ps_weights = sm2np(g_ps_weights_sm, (self.params["grad_size"],))
        ps_weights[:] = new_ps_weights

        if self.params['do_virtual_momentum_sketch']:
            self.momentums = new_momentums
        elif self.params['do_local_momentum_sketch']:
            for i, idx in enumerate(indices):
                self.momentums[idx] = new_momentums[i]
        if self.params['do_virtual_error_sketch']:
            self.errors = new_errors
        elif self.params['do_local_error_sketch']:
            for i, idx in enumerate(indices):
                self.errors[idx] = new_errors[i]
        #self.errors = new_errors
        #client_weights = update_params(client_weights, indices, new_momentums, MOMENTUM_ID)
        #client_weights = update_params(client_weights, indices, new_errors, ERROR_ID)
        #"""

    def zero_grad(self):
        pass
class FedCommEffCriterion:
    def __init__(self, input_criterion, params):
        global g_criterion
        g_criterion = input_criterion
    def __call__(self, *args):
        global g_criterion
        out = g_criterion(*args)
        return out
class FedCommEffAccuracy:
    def __init__(self, input_accuracy, params):
        global g_accuracy
        g_accuracy = input_accuracy
    def __call__(self, *args):
        global g_accuracy
        out = g_accuracy(*args)
        return out

#@ray.remote(num_gpus=GPUS_ALLOCATED, num_return_vals=5, num_cpus=CPUS_ALLOCATED)
def update_forward_grad(client_weights, client_error, curr_weights, model_cls,
        model_config, ins, targets, params):
        #criterion, accuracy, params):
    global g_criterion
    global g_accuracy
    new_client_weights = client_update(client_weights, curr_weights, params)
    outs, loss, acc, grad = forward_grad(model_cls, model_config,
            new_client_weights, client_error, ins, targets, g_criterion,
            g_accuracy, params)
    return outs, loss, acc, grad, new_client_weights.cpu()

def get_worker_device(params):
    # cpu => cpu; cuda => cuda:#
    # bad! relying on private _identity. Lazy!
    device = params["device"]
    n_workers = params["n_workers"]
    worker_id = multiprocessing.current_process()._identity[0]
    if device == "cuda":
        device = "cuda:{:d}".format(worker_id % n_workers)
        #device = "cuda:{:d}".format((worker_id % 7) + 1)
    return device

def update_forward_grad_sketched(worker_id, client_id, model_cls,
                                 model_config, ins, targets, params,
                                 criterion, accuracy):

    # pull PS and client weights out of the shared memory block
    grad_size = params["grad_size"]
    n_clients = params["n_clients"]
    participation = params["participation"]

    global gw_worker_Sgrads_sm
    global gw_client_weights_sm
    global gw_ps_weights_sm

    ps_weights = sm2np(gw_ps_weights_sm, (grad_size,))
    client_weights = sm2np(gw_client_weights_sm, (n_clients, grad_size))

    worker_weights = client_weights[client_id]


    params["device"] = get_worker_device(params)
    ps_weights = torch.from_numpy(ps_weights).to(params["device"])
    worker_weights = torch.from_numpy(worker_weights).to(params["device"])


    new_worker_weights = get_new_worker_weights(ps_weights,
                                                worker_weights,
                                                params)

    outs, loss, acc, Sgrad = forward_grad_sketched(
            model_cls, model_config, new_worker_weights,
            ins, targets, criterion, accuracy, params
        )

    # write grad to the shared memory grad array in spot worker_id
    n_workers = int(n_clients * participation)
    worker_Sgrads_shape = (n_workers, params["num_rows"], params["num_cols"])
    worker_Sgrads = sm2np(gw_worker_Sgrads_sm, worker_Sgrads_shape)
    worker_Sgrads[worker_id,:,:] = Sgrad.cpu().numpy()[:,:]

    return outs.numpy(), loss, acc

def get_new_worker_weights(ps_weights, worker_weights, params):
    device = params['device']

    ps_weights = ps_weights.to(device)
    worker_weights = worker_weights.to(device)

    # we'll update the old worker_weights with a possibly compressed
    # version of diff_vec
    diff_vec = ps_weights - worker_weights
    topk_down = params['topk_down']
    """
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
    """
    if topk_down:
        k = params['k']
        weight_update = _topk(diff_vec, k=k)
    else:
        weight_update = diff_vec

    new_worker_weights = worker_weights + weight_update
    #print(f"{torch.norm(weight_update, 2)}")
    #print(f"{updated_vec} = {client_weights} + {weight_update}")
    return new_worker_weights
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

def forward_grad_sketched(model_cls, model_config, weights, ins, targets,
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
    table = sketch.table.cpu()
    del sketch
    return outs.cpu().detach(), loss, acc, table

#@ray.remote(num_gpus=GPUS_ALLOCATED, num_return_vals=3)
def forward(model_cls, model_config, ins, targets,
        params):
        #criterion, accuracy, params):
    global g_criterion
    global g_accuracy

    device = params['device']

    global g_ps_weights_sm
    ps_weights = sm2np(g_ps_weights_sm, (params["grad_size"],))
    weights = torch.from_numpy(ps_weights).to(device)

    model = model_cls(**model_config).to(device)
    weights = weights.to(device)
    set_param_vec(model, weights)
    ins = ins.to(device)
    targets = targets.to(device)
    criterion = g_criterion.to(device)
    accuracy = g_accuracy.to(device)
    outs = model(ins)
    loss, acc = compute_loss(outs, targets, criterion, accuracy, train=False)
    return outs.cpu().detach(), loss, acc

def get_updated_server_sketched(momentum_sketches, error_sketches, params, lr, sketch):
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

    global g_ps_weights_sm
    global g_client_weights_sm
    global g_worker_Sgrads_sm
    ps_weights = sm2np(g_ps_weights_sm, (params["grad_size"],),)
    client_weights = sm2np(g_client_weights_sm,
                           (params["n_clients"], params["grad_size"]))
    n_workers = int(params["n_clients"] * params["participation"])
    worker_Sgrads_shape = (n_workers, params["num_rows"], params["num_cols"])
    worker_Sgrads = sm2np(g_worker_Sgrads_sm, worker_Sgrads_shape)

    ps_weights = torch.from_numpy(ps_weights).to(device)

    worker_Sgrads = [torch.from_numpy(Sg).to(device)
                     for Sg in worker_Sgrads]

    for grad_sketch, momentum_sketch, error_sketch in zip(worker_Sgrads, momentum_sketches, error_sketches):
        if params['do_virtual_momentum_sketch'] or params['do_local_momentum_sketch']:
            momentum_sketch *= momentum
            momentum_sketch.accumulateTable(grad_sketch)
            if params['do_virtual_error_sketch'] or params['do_local_error_sketch']:
                error_sketch += momentum_sketch
        elif params['do_virtual_error_sketch'] or params['do_local_error_sketch']:
            error_sketch.accumulateTable(grad_sketch)
        else:
            sketch.accumulateTable(grad_sketch)
    if params['do_virtual_error_sketch'] or params['do_local_error_sketch']:
        update = np.sum(error_sketches).unSketch(k=k)
    else:
        update = sketch.unSketch(k=k)
    sketch.zero()
    sketch.accumulateVec(update)
    hh_coords = sketch.table.nonzero()
    hh_0, hh_1 = hh_coords[:, 0], hh_coords[:, 1]
    for momentum_sketch, error_sketch in zip(momentum_sketches, error_sketches):
        if params['do_virtual_momentum_sketch'] or params['do_local_momentum_sketch']:
            momentum_sketch.table[hh_0, hh_1] = 0
        if params['do_virtual_error_sketch'] or params['do_local_error_sketch']:
            error_sketch.table[hh_0, hh_1] = 0
    weight_update = update * lr
    updated_weights = ps_weights - weight_update
    #print(f"{updated_weights} = {curr_weights} - {weight_update} ({update} * {lr}) from {grads}")
    return updated_weights.cpu(), momentum_sketches, error_sketches

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

def get_weights(client_weights, idx):
    retval = client_weights[idx][WEIGHT_ID]
    return retval

def get_error(client_weights, idx):
    retval = client_weights[idx][ERROR_ID]
    return retval

def get_errors(client_weights, indices):
    return [client_weights[idx][ERROR_ID] for idx in indices]

def get_params(indices, idx):
    return [client_weights[index][idx] for index in indices]

def update_params(client_weights, client_indices, vecs, vec_idx):
    for i, idx in enumerate(client_indices):
        client_weights[idx][vec_idx] = vecs[i]
    return client_weights

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

def get_grad(model, weights, params, train=True, device='cpu'):
    if train:
        grad_vec = get_grad_vec(model)
        if params['weight_decay'] != 0:
            grad_vec.add_(params['weight_decay']/params['n_workers'],
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

def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

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

# GPT2 FUNCTIONS
class FedCommEffModelGPT2:
    def __init__(self, input_model, model_cls, params):
        n_clients = params['n_clients']
        participation = params["participation"]
        device = params['device']
        cpu = "cpu"
        self.model_cls = model_cls
        self.model = input_model.to(device)
        param_vec = get_param_vec(self.model, cpu).numpy()
        grad_size = 0
        for p in self.model.parameters():
            if p.requires_grad:
                grad_size += torch.numel(p)

        global g_worker_Sgrads_sm
        global g_client_weights_sm
        global g_ps_weights_sm

        # ps_weights needs to be in shared memory so the workers can
        # update themselves with (possibly an approximation of) the
        # latest PS weights

        g_ps_weights_sm = Array(ctypes.c_float,
                                param_vec.size,
                                lock=False)
        ps_weights = sm2np(g_ps_weights_sm, param_vec.shape)
        # store the initial weights of the model
        ps_weights[:] = param_vec[:]

        # client weights emulates each client's possibly stale weights
        g_client_weights_sm = Array(ctypes.c_float,
                                    param_vec.size * n_clients,
                                    lock=False)

        # copy ps_weights into every row of client_weights
        client_weights = sm2np(g_client_weights_sm,
                               (n_clients, param_vec.size))
        client_weights[:] = np.tile(param_vec, (n_clients, 1))

        # this shared memory block will hold the gradient sketches computed
        # by each worker in a round
        n_workers = params['n_workers']
        g_worker_Sgrads_sm = Array(
                'f',
                n_workers * params["num_rows"] * params["num_cols"],
                lock=False
            )
        # and zero out worker_Sgrads to start
        worker_Sgrads_shape = (n_workers, params["num_rows"], params["num_cols"])
        worker_Sgrads = sm2np(g_worker_Sgrads_sm, worker_Sgrads_shape)
        worker_Sgrads[:] = 0

        self.process_pool = multiprocessing.Pool(
                params['n_workers'],
                initializer=init_pool,
                initargs=(g_worker_Sgrads_sm, g_client_weights_sm,
                          g_ps_weights_sm)
            )

        self.params = params
        self.params['grad_size'] = grad_size

    def train(self, training):
        self.training = training
    def save_pretrained(self, log_dir):
        global g_ps_weights_sm
        ps_weights = sm2np(g_ps_weights_sm,
                           (self.params["grad_size"],))
        curr_weights = torch.from_numpy(ps_weights)
        curr_weights = curr_weights.to(self.params['device'])
        set_param_vec(self.model, curr_weights)
        self.model.save_pretrained(log_dir)
    def __call__(self, batches, indices):
        client_params = self.client_params
        if self.training:
            args_tuples = [(i, idx, self.model, batches[i], self.params)
                            for i, idx in enumerate(indices)]
            results = self.process_pool.starmap(update_forward_grad_sketched_gpt2,
                                                args_tuples)
            #loss = np.array([r[0] for r in results])
            return results

        else:
            global g_criterion
            """
            args_tuples = [(self.model, batches[i], self.params, g_criterion)
                            for i, idx in enumerate(indices)]
            results = self.process_pool.starmap(forward_gpt2, args_tuples)
            results = [forward_gpt2(self.model, batches[i], self.params, g_criterion)
                    for i, idx in enumerate(indices)]
            nlls = np.array([r[0] for r in results])
            accs = np.array([r[1] for r in results])
            ppls = np.array([r[2] for r in results])
            """
            nlls, accs, ppls = forward_gpt2(self.model, batches, self.params, g_criterion)
            return nlls, accs, ppls

    def __getattr__(self, name):
        if name == "parameters":
            global g_ps_weights_sm
            ps_weights = sm2np(g_ps_weights_sm,
                               (self.params["grad_size"],))
            curr_weights = torch.from_numpy(ps_weights)
            curr_weights = curr_weights.to(self.params['device'])
            set_param_vec(self.model, curr_weights)
            return getattr(self.model, name)

def update_forward_grad_sketched_gpt2(worker_id, client_id, model,
                                 batch, params):

    # pull PS and client weights out of the shared memory block
    grad_size = params["grad_size"]
    n_clients = params["n_clients"]
    participation = params["participation"]

    global gw_worker_Sgrads_sm
    global gw_client_weights_sm
    global gw_ps_weights_sm

    ps_weights = sm2np(gw_ps_weights_sm, (grad_size,))
    client_weights = sm2np(gw_client_weights_sm, (n_clients, grad_size))

    worker_weights = client_weights[client_id]


    params["device"] = get_worker_device(params)
    ps_weights = torch.from_numpy(ps_weights).to(params["device"])
    worker_weights = torch.from_numpy(worker_weights).to(params["device"])
    model.to(params["device"])

    new_worker_weights = get_new_worker_weights(ps_weights,
                                                worker_weights,
                                                params)

    loss, Sgrad = forward_grad_sketched_gpt2(
            model, new_worker_weights,
            batch, params
        )

    # write grad to the shared memory grad array in spot worker_id
    n_workers = params["n_workers"]
    worker_Sgrads_shape = (n_workers, params["num_rows"], params["num_cols"])
    worker_Sgrads = sm2np(gw_worker_Sgrads_sm, worker_Sgrads_shape)
    worker_Sgrads[worker_id,:,:] = Sgrad.cpu().numpy()[:,:]

    return loss

def forward_grad_sketched_gpt2(model, weights, batch, params):
    device = params['device']
    #model.to(device)
    model.train()
    #weights = weights.to(device)
    set_param_vec(model, weights)
    mega_batch = batch
    grad_accum_steps = params['grad_accum_steps']
    batch_size = params['batch_size']
    #train_batch_size = params['train_batch_size']
    #print(f"Real batch size: {mega_batch[3].size()[0]} from {train_batch_size} with {[b.size() for b in batch]}")
    train_batch_size = mega_batch[3].size()[0]
    accum_loss = None
    n_steps = train_batch_size // batch_size
    for i in range(n_steps):
        start = i * batch_size
        end = (i+1) * batch_size
        batch = [b[start:end] for b in mega_batch]
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        (lm_loss), (mc_loss), *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, lm_labels=lm_labels
        )
        loss = (lm_loss * params['lm_coef'] + mc_loss * params['mc_coef']) / n_steps
        print(f"Loss: {loss} from {lm_loss} and {mc_loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_norm'])
        if accum_loss is not None:
            accum_loss += loss
        else:
            accum_loss = loss
    grad = get_grad(model, weights, params, device=device)
    sketch = CSVec(d=params['grad_size'], c=params['num_cols'],
        r=params['num_rows'], device=device,
        numBlocks=params['num_blocks'])
    sketch.accumulateVec(grad)
    table = sketch.table.cpu()
    return accum_loss.item()/max(n_steps, 1), table

def forward_gpt2(model, batch, params, criterion):
    # pull PS weights out of the shared memory block
    grad_size = params["grad_size"]
    device = params['device']
    global g_ps_weights_sm
    ps_weights = sm2np(g_ps_weights_sm, (params["grad_size"],))
    ps_weights = torch.from_numpy(ps_weights).to(params["device"])
    model.to(device)
    model.eval()
    set_param_vec(model, ps_weights)
    logits, labels = inference(model, batch, params)
    lm_logits, mc_logits = logits
    lm_labels, mc_labels = labels
    nll = criterion(lm_logits, lm_labels).detach().cpu().numpy()
    acc = accuracy(mc_logits, mc_labels)
    ppl = np.exp(nll)
    return nll, acc, ppl

def accuracy(y_pred, y):
    y_pred, y = _check_shape(y_pred, y)
    indices = torch.argmax(y_pred, dim=1)
    correct = torch.eq(indices, y).view(-1)
    _num_correct, _num_examples = 0, 0
    _num_correct += torch.sum(correct).item()
    _num_examples += correct.shape[0]
    acc = _num_correct/_num_examples
    return acc

def _check_shape(y_pred, y):
    if y.ndimension() > 1 and y.shape[1] == 1:
        # (N, 1, ...) -> (N, ...)
        y = y.squeeze(dim=1)
    if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
        # (N, 1, ...) -> (N, ...)
        y_pred = y_pred.squeeze(dim=1)
    if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
        raise ValueError("y must have shape of (batch_size, ...) and y_pred must have "
             "shape of (batch_size, num_categories, ...) or (batch_size, ...), "
             "but given {} vs {}.".format(y.shape, y_pred.shape))
    y_shape = y.shape
    y_pred_shape = y_pred.shape
    if y.ndimension() + 1 == y_pred.ndimension():
        y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]
    if not (y_shape == y_pred_shape):
        raise ValueError("y and y_pred must have compatible shapes.")
    return y_pred, y

def inference(model, batch, params):
    model.eval()
    with torch.no_grad():
        batch = tuple(input_tensor.to(params["device"]) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        model_outputs = model(input_ids, mc_token_ids, token_type_ids=token_type_ids)
        lm_logits, mc_logits, *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
        )
        lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
        return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
