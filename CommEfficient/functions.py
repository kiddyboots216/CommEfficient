import torch
from datetime import datetime
import os
import numpy as np
from csvec import CSVec
import copy

import cProfile

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
g_worker_grads_sm = None
g_client_weights_sm = None
g_ps_weights_sm = None

g_criterion = None
g_accuracy = None

def make_logdir(params: dict):
    rows = params["num_rows"]
    cols = params["num_cols"]
    k = params["k"]
    mode = params["mode"]
    local_iters = params.get("n_local_iters", None)
    sketch_str = f"{mode}: {rows} x {cols}" if mode == "sketch" else "{mode}"
    k_str = f"k: {k}" if mode in ["sketch", "true_topk", "local_topk"] else f"local_iters: {local_iters}"
    workers = params["n_workers"]
    clients = params["n_clients"]
    clients_str = f"{workers}/{clients}"
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + clients_str + '_' + sketch_str + '_' + k_str)
    return logdir

def init_pool(worker_Sgrads_sm, worker_grads_sm,
              client_weights_sm, ps_weights_sm):
    global gw_ps_weights_sm
    global gw_client_weights_sm
    global gw_worker_Sgrads_sm
    global gw_worker_grads_sm

    gw_ps_weights_sm = ps_weights_sm
    gw_client_weights_sm = client_weights_sm
    gw_worker_Sgrads_sm = worker_Sgrads_sm
    gw_worker_grads_sm = worker_grads_sm

def sm2np(sm, shape, dtype=ctypes.c_float):
    # convert from shared memory object/buffer to numpy array
    nparray = np.ndarray(shape, dtype=dtype, buffer=sm)
    assert(nparray.base is sm)
    return nparray

def profile_helper(*args):
    cProfile.runctx("update_forward_grad(*args)",
                    globals(), locals(),
                    "profile/init{:d}.prof".format(
                        multiprocessing.current_process()._identity[0]
                    )
                   )

class FedCommEffModel:
    def __init__(self, input_model, params):
        n_clients = params['n_clients']
        participation = params["participation"]
        device = params['device']
        cpu = "cpu"
        if params.get('unit_test', False):
            list(input_model.parameters())[0].data.zero_()
        self.model = input_model.to(device)
        param_vec = get_param_vec(self.model, cpu).numpy()
        grad_size = 0
        for p in self.model.parameters():
            if p.requires_grad:
                grad_size += torch.numel(p)
        params['grad_size'] = grad_size
        self.params = params

        global g_ps_weights_sm
        global g_client_weights_sm
        global g_worker_Sgrads_sm
        global g_worker_grads_sm

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

        # this shared memory block will hold the gradients
        # (or gradient sketches) computed by each worker in a round
        n_workers = params['n_workers']
        if params["mode"] == "sketch":
            g_worker_Sgrads_sm = Array(
                    'f',
                    n_workers * params["num_rows"] * params["num_cols"],
                    lock=False
                )
            # and zero out worker_Sgrads to start
            worker_Sgrads_shape = (n_workers,
                                   params["num_rows"],
                                   params["num_cols"])
            worker_Sgrads = sm2np(g_worker_Sgrads_sm, worker_Sgrads_shape)
            worker_Sgrads[:] = 0
        elif params["mode"] in ["true_topk", "local_topk", "localSGD"]:
            g_worker_grads_sm = Array(
                    'f',
                    n_workers * params["grad_size"],
                    lock=False
                )
            worker_grads_shape = (n_workers, params["grad_size"])
            worker_grads = sm2np(g_worker_grads_sm, worker_grads_shape)
            worker_grads[:] = 0

        # process pool that parallelizes training
        self.process_pool = multiprocessing.Pool(
                n_workers,
                initializer=init_pool,
                initargs=(g_worker_Sgrads_sm, g_worker_grads_sm, 
                          g_client_weights_sm, g_ps_weights_sm)
            )

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
        global g_criterion
        global g_metric
        #global lr
        if self.training:
            #self.params["lr"] = lr
            args_tuples = [(i, idx, self.model,
                            batches[i], self.params,
                            g_criterion, g_metric)
                           for i, idx in enumerate(indices)]
            results = self.process_pool.starmap(
                    #profile_helper,
                    update_forward_grad,
                    args_tuples
                )
            return split_results(results, self.params["n_results_train"])

        else:
            args_tuples = [(self.model,
                batches[i], self.params, g_criterion, g_metric)
                for i, idx in enumerate(indices)]
            results = self.process_pool.starmap(forward_multiprocessed,
                    args_tuples)
            return split_results(results, self.params["n_results_val"])

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

        # helper for rest of __init__
        def initialize_helper(thing_type, base_thing):
            if thing_type == "virtual":
                return [copy.deepcopy(base_thing)]
            elif thing_type == "local":
                return [copy.deepcopy(base_thing)
                        for _ in range(params["n_clients"])]
            elif thing_type == "none":
                return None
            else:
                msg = "{} is an invalid type"
                raise ValueError(msg.format(thing_type))

        # hack so there isn't a not defined error later...
        self.base_sketch = None
        if params["mode"] == "sketch":
            sketch = CSVec(d=grad_size, c=params['num_cols'],
                           r=params['num_rows'], device=device,
                           numBlocks=params['num_blocks'])
            # base_sketch can be used for miscellaneous
            # sketching activities
            self.base_sketch = sketch

            # create momentum & error sketches -- one or one for each
            # client depending on whether we're doing virtual momentum
            self.momentums = initialize_helper(
                    params["momentum_type"], sketch
                )
            self.errors = initialize_helper(
                    params["error_type"], sketch
                )
        elif params["mode"] in ["true_topk", "local_topk"]:
            # same as above but with vectors instead of sketches
            zero_vec = torch.zeros(params["grad_size"]).to(device)
            self.momentums = initialize_helper(
                    params["momentum_type"], zero_vec
                )
            self.errors = initialize_helper(
                    params["error_type"], zero_vec
                )
        elif params["mode"] == "localSGD":
            self.momentums = initialize_helper("none", None)
            self.errors = initialize_helper("none", None)

    def get_lr(self):
        new_lr = get_lr(self.param_groups)
        """
        if self.params["mean_grads"]:
            new_lr = new_lr / self.params["n_workers"]
        """
        global lr
        lr = new_lr
        return new_lr

    def step(self, indices, ret=False):
        # in this method we're agnostic as to whether we're sketched,
        # true topk, or local topk
        lr = self.get_lr()
        if ret:
            return
        cur_momentums = None
        if self.params['momentum_type'] == "virtual":
            cur_momentums = self.momentums
        elif self.params['momentum_type'] == "local":
            cur_momentums = [self.momentums[idx] for idx in indices]
        elif self.params["momentum_type"] == "none":
            cur_momentums = None
        else:
            msg = "invalid momentum type {}"
            raise ValueError(msg.format(self.params["momentum_type"]))

        cur_errors = None
        if self.params['error_type'] == "virtual":
            cur_errors = self.errors
        elif self.params['error_type'] == "local":
            cur_errors = [self.errors[idx] for idx in indices]
        elif self.params["error_type"] == "none":
            cur_errors = None
        else:
            msg = "invalid error type {}"
            raise ValueError(msg.format(self.params["error_type"]))

        new_ps_weights, new_momentums, new_errors = get_updated_server(
                cur_momentums,
                cur_errors,
                self.params,
                lr,
                self.base_sketch)

        # update ps_weights, momentums, and errors
        ps_weights = sm2np(g_ps_weights_sm, (self.params["grad_size"],))
        ps_weights[:] = new_ps_weights

        if self.params['momentum_type'] == "virtual":
            self.momentums = new_momentums
        elif self.params['momentum_type'] == "local":
            for i, idx in enumerate(indices):
                self.momentums[idx] = new_momentums[i]

        if self.params['error_type'] == "virtual":
            self.errors = new_errors
        elif self.params['error_type'] == "local":
            for i, idx in enumerate(indices):
                self.errors[idx] = new_errors[i]

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
class FedCommEffMetric:
    def __init__(self, input_metric, params):
        global g_metric
        g_metric = input_metric
    def __call__(self, *args):
        global g_metric
        out = g_metric(*args)
        return out

def get_worker_device(params):
    # cpu => cpu; cuda => cuda:#
    # bad! relying on private _identity. Lazy!
    device = params["device"]
    n_workers = params["n_workers"]
    large = params["model"] == "gpt2"
    worker_id = multiprocessing.current_process()._identity[0]
    if device == "cuda":
        device = "cuda:{:d}".format(worker_id % n_workers)
        if large:
            device = "cuda:{:d}".format((worker_id % n_workers) + 1)
    return device

def update_forward_grad(worker_id, client_id, model,
                        batch, params, criterion, metric):

    # pull PS and client weights out of the shared memory block
    grad_size = params["grad_size"]
    n_clients = params["n_clients"]
    participation = params["participation"]

    global gw_ps_weights_sm
    global gw_client_weights_sm
    global gw_worker_Sgrads_sm
    global gw_worker_grads_sm

    ps_weights = sm2np(gw_ps_weights_sm, (grad_size,))
    client_weights = sm2np(gw_client_weights_sm, (n_clients, grad_size))
    worker_weights = client_weights[client_id]
    params["device"] = get_worker_device(params)
    ps_weights = torch.from_numpy(ps_weights).to(params["device"])
    worker_weights = torch.from_numpy(worker_weights).to(params["device"])

    new_worker_weights = get_new_worker_weights(ps_weights,
                                                worker_weights,
                                                params)

    # g is a (possibly compressed) gradient
    f = forward_grad
    if params["model"] == "gpt2":
        f = forward_grad_gpt2
    g, results = f(
            model, new_worker_weights,
            batch, criterion, metric, params
        )
    """
    g, grad, results = f(
            model, new_worker_weights,
            batch, criterion, metric, params
        )
    """
    # write g to the shared memory grad array in spot worker_id
    n_workers = int(n_clients * participation)
    if params["mode"] == "sketch":
        worker_Sgrads_shape = (n_workers, params["num_rows"], params["num_cols"])
        worker_Sgrads = sm2np(gw_worker_Sgrads_sm, worker_Sgrads_shape)
        worker_Sgrads[worker_id,:,:] = g.cpu().numpy()[:,:]
    elif params["mode"] in ["true_topk", "local_topk", "localSGD"]:
        worker_grads_shape = (n_workers, params["grad_size"])
        worker_grads = sm2np(gw_worker_grads_sm, worker_grads_shape)
        worker_grads[worker_id,:] = g.cpu().numpy()[:]

    return results

def get_new_worker_weights(ps_weights, worker_weights, params):
    device = params['device']

    ps_weights = ps_weights.to(device)
    worker_weights = worker_weights.to(device)

    # we'll update the old worker_weights with a possibly compressed
    # version of diff_vec
    diff_vec = ps_weights - worker_weights
    if params['topk_down']:
        weight_update = _topk(diff_vec, k=params["k"])
    else:
        weight_update = diff_vec

    new_worker_weights = worker_weights + weight_update
    #print(f"{torch.norm(weight_update, 2)}")
    #print(f"{updated_vec} = {client_weights} + {weight_update}")
    return new_worker_weights

def forward_grad(model, weights, batch,
                 criterion, metric, params):
    device = params['device']
    model = model.to(device)
    model.train()
    weights = weights.to(device)
    set_param_vec(model, weights)
    batch = tuple(input_tensor.to(device) for input_tensor in batch)
    criterion = criterion.to(device)
    metric = metric.to(device)
    if params["supervised"]:
        ins, targets = batch
        outs = model(ins)
        results = compute_loss(outs, targets, criterion, metric, train=True)
    grad = get_grad(model, weights, params, train=True, device=device)

    # compress the gradient if needed
    if params["mode"] == "sketch":
        sketch = CSVec(d=params['grad_size'], c=params['num_cols'],
            r=params['num_rows'], device=device,
            numBlocks=params['num_blocks'])
        sketch.accumulateVec(grad)
        g = sketch.table.cpu()
        del sketch
    elif params["mode"] == "true_topk":
        g = grad
    elif params["mode"] == "local_topk":
        g = _topk(grad, k=params["k"])
    elif params["mode"] == "localSGD":
        grad *= params["lr"]
        weights -= grad
        params["n_local_iters"] -= 1
        if params["n_local_iters"] > 0:
            g_recursive, results_recursive = forward_grad(model, weights, batch,
                    criterion, metric, params)
            g = grad + g_recursive
            results = [r + r_recursive for (r, r_recursive) in zip(results, results_recursive)]
        else:
            g = grad

    return g, results

def forward_multiprocessed(model, batch,
        params, criterion, metric):
    grad_size = params["grad_size"]
    global gw_ps_weights_sm
    ps_weights = sm2np(gw_ps_weights_sm, (grad_size,))
    params["device"] = get_worker_device(params)
    device = params["device"]
    ps_weights = torch.from_numpy(ps_weights).to(params["device"])
    model = model.to(device)
    model.eval()
    set_param_vec(model, ps_weights)
    batch = tuple(input_tensor.to(device) for input_tensor in batch)
    if params.get("supervised", None):
        criterion = criterion.to(device)
        metric = metric.to(device)
        ins, targets = batch
        outs = model(ins)
        results = compute_loss(outs, targets, criterion, metric, train=False)
    else:
        logits, labels = inference(model, batch, params)
        lm_logits, mc_logits = logits
        lm_labels, mc_labels = labels
        nll = criterion(lm_logits, lm_labels).detach().cpu().numpy()
        acc = accuracy(mc_logits, mc_labels)
        results = nll, acc
    return results

def get_updated_server(momentums, errors, params, lr, sketch=None):
    if params["mode"] == "sketch":
        return _server_helper_sketched(momentums, errors, params,
                                       lr, sketch)
    elif params["mode"] == "local_topk":
        return _server_helper_local_topk(momentums, errors, params, lr)
    elif params["mode"] == "true_topk":
        return _server_helper_true_topk(momentums, errors, params, lr)
    elif params["mode"] == "localSGD":
        return _server_helper_localSGD(momentums, errors, params, lr)
    else:
        assert False, "invalid mode {}".format(params["mode"])

def _server_helper_localSGD(momentum_vecs, error_vecs, params, lr):
    global ps_weights_sm
    global p_worker_grads_sm
    device = torch.device(params['device'])
    ps_weights = sm2np(g_ps_weights_sm, (params["grad_size"],),)
    ps_weights = torch.from_numpy(ps_weights).to(device)
    n_workers = int(params["n_clients"] * params["participation"])
    worker_grads_shape = (n_workers, params["grad_size"])
    worker_grads = sm2np(g_worker_grads_sm, worker_grads_shape)
    large = params["model"] == "gpt2"
    if large:
        grad_sum = torch.from_numpy(worker_grads[0]).to(device)
        for g in worker_grads[1:]:
            grad_sum += torch.from_numpy(g).to(device)
    else:
        grad_sum = np.sum([torch.from_numpy(g).to(device) for g in worker_grads])
    update = grad_sum
    return (ps_weights - update).cpu(), momentum_vecs, error_vecs

def _server_helper_true_topk(momentum_vecs, error_vecs, params, lr):
    global g_ps_weights_sm
    global g_worker_grads_sm
    assert params["momentum_type"] == "virtual"
    assert params["error_type"] == "virtual"

    device = torch.device(params['device'])
    momentum = params['momentum']

    ps_weights = sm2np(g_ps_weights_sm, (params["grad_size"],),)
    ps_weights = torch.from_numpy(ps_weights).to(device)

    n_workers = int(params["n_clients"] * params["participation"])
    worker_grads_shape = (n_workers, params["grad_size"])
    worker_grads = sm2np(g_worker_grads_sm, worker_grads_shape)
    large = params["model"] == "gpt2"
    if large:
        grad_sum = torch.from_numpy(worker_grads[0]).to(device)
        for g in worker_grads[1:]:
            grad_sum += torch.from_numpy(g).to(device)
    else:
        grad_sum = np.sum([torch.from_numpy(g).to(device) for g in worker_grads])
    momentum_vec = momentum_vecs[0]
    error_vec = error_vecs[0]
    momentum_vec *= momentum
    momentum_vec += grad_sum
    error_vec += momentum_vec
    update = _topk(error_vec, k=params["k"])
    momentum_vec[update.nonzero()] = 0
    error_vec[update.nonzero()] = 0
    return (ps_weights - update * lr).cpu(), [momentum_vec], [error_vec]

def _server_helper_local_topk(momentum_vecs, error_vecs, params, lr):
    global g_ps_weights_sm
    global g_worker_grads_sm
    assert params["momentum_type"] == "virtual"
    assert params["error_type"] == "virtual"

    device = torch.device(params['device'])
    momentum = params['momentum']

    ps_weights = sm2np(g_ps_weights_sm, (params["grad_size"],),)
    ps_weights = torch.from_numpy(ps_weights).to(device)

    n_workers = int(params["n_clients"] * params["participation"])
    worker_grads_shape = (n_workers, params["grad_size"])
    worker_grads = sm2np(g_worker_grads_sm, worker_grads_shape)
    large = params["model"] == "gpt2"
    if large:
        grad_sum = torch.from_numpy(worker_grads[0]).to(device)
        for g in worker_grads[1:]:
            grad_sum += torch.from_numpy(g).to(device)
    else:
        grad_sum = np.sum([torch.from_numpy(g).to(device) for g in worker_grads])
    momentum_vec = momentum_vecs[0]
    error_vec = error_vecs[0]
    momentum_vec *= momentum
    momentum_vec += grad_sum
    error_vec += momentum_vec
    update = _topk(error_vec, k=params["k"])
    momentum_vec[update.nonzero()] = 0
    error_vec[update.nonzero()] = 0
    return (ps_weights - update * lr).cpu(), [momentum_vec], [error_vec]

def _server_helper_sketched(momentum_sketches, error_sketches,
                            params, lr, sketch):
    momentum = params['momentum']
    k = params['k']
    device = torch.device(params['device'])
    momentum_type = params['momentum_type']
    error_type = params['error_type']
    local = momentum_type == 'local' and error_type == 'local'
    virtual = momentum_type == 'virtual' and error_type == 'virtual'
    none = momentum_type == 'none' and error_type == 'none'
    assert local or virtual or none

    global g_ps_weights_sm
    global g_worker_Sgrads_sm

    ps_weights = sm2np(g_ps_weights_sm, (params["grad_size"],),)
    ps_weights = torch.from_numpy(ps_weights).to(device)

    n_workers = int(params["n_clients"] * params["participation"])
    worker_Sgrads_shape = (n_workers,
                           params["num_rows"],
                           params["num_cols"])
    worker_Sgrads = sm2np(g_worker_Sgrads_sm, worker_Sgrads_shape)
    worker_Sgrads = [torch.from_numpy(Sg).to(device)
                     for Sg in worker_Sgrads]
    if local:
        for grad_table, momentum_sketch, error_sketch in zip(worker_Sgrads, momentum_sketches, error_sketches):
            if params['momentum_type'] != "none":
                momentum_sketch *= momentum
                momentum_sketch.accumulateTable(grad_table)
                if params['error_type'] != "none":
                    error_sketch += momentum_sketch
            elif params["error_type"] != "none":
                error_sketch.accumulateTable(grad_table)
            else:
                sketch += grad_sketch
        if params['error_type'] != "none":
            update = np.sum(error_sketches).unSketch(k=k)
        else:
            update = sketch.unSketch(k=k)
        sketch.zero()
        sketch.accumulateVec(update)
        hh_coords = sketch.table.nonzero()
        hh_0, hh_1 = hh_coords[:, 0], hh_coords[:, 1]
        for momentum_sketch, error_sketch in zip(momentum_sketches, error_sketches):
            if params["momentum_type"] != "none":
                momentum_sketch.table[hh_0, hh_1] = 0
            if params['error_type'] != "none":
                error_sketch.table[hh_0, hh_1] = 0

    elif virtual:
        grad_sketch_sum = sketch
        grad_sketch_sum.zero()
        for S in worker_Sgrads:
            grad_sketch_sum.accumulateTable(S)

        momentum_sketch = momentum_sketches[0]
        error_sketch = error_sketches[0]
        if params['momentum_type'] != "none":
            momentum_sketch *= momentum
            momentum_sketch += grad_sketch_sum
            if params["error_type"] != "none":
                error_sketch += momentum_sketch
        elif params["error_type"] != "none":
            error_sketch += grad_sketch_sum
        else:
            sketch += grad_sketch_sum
        if params['error_type'] != "none":
            update = error_sketch.unSketch(k=k)
        elif params['momentum_type'] != "none":
            update = momentum_sketch.unSketch(k=k)
        else:
            update = sketch.unSketch(k=k)
        sketch.zero()
        sketch.accumulateVec(update)
        hh_coords = sketch.table.nonzero()
        hh_0, hh_1 = hh_coords[:, 0], hh_coords[:, 1]
        if params["momentum_type"] != "none":
            momentum_sketch.table[hh_0, hh_1] = 0
        if params['error_type'] != "none":
            error_sketch.table[hh_0, hh_1] = 0
        momentum_sketches = [momentum_sketch]
        error_sketches = [error_sketch]
    
    elif none:
        global g_worker_grads_sm
        worker_grads_shape = (n_workers, params["grad_size"])
        worker_grads = sm2np(g_worker_grads_sm, worker_grads_shape)
        grad_sum = np.sum(worker_grads)
        grad_sketch_sum = sketch
        grad_sketch_sum.zero()
        for S in worker_Sgrads:
            grad_sketch_sum.accumulateTable(S)
        update = grad_sketch_sum.unSketch(k=k)
        print(f"Reconstruction error: {(update - grad_sum).norm()}")
    return (ps_weights - update * lr).cpu(), momentum_sketches, error_sketches

def get_lr(optimizer_param_groups):
    if len(optimizer_param_groups) == 1:
        lr = optimizer_param_groups[0]["lr"]
        #print(f"Lr is {lr}")
        return lr

def _topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    ret = torch.zeros_like(vec)
    # on a gpu, sorting is faster than pytorch's topk method
    #topkIndices = torch.sort(vec**2)[1][-k:]
    # however, torch.topk is more space efficient
    topkIndices = torch.topk(vec**2, k, sorted=False)[1]
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

def compute_loss(outs, targets, criterion, metric, train):
    loss = criterion(outs, targets)
    if train:
        loss.backward()
    batch_loss = loss.mean().cpu().detach().numpy()
    acc = metric(outs, targets).float().mean().cpu().detach().numpy()
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

def split_results(results, n_results):
    return [np.array([r[i] for r in results]) for i in range(n_results)]    

# GPT2 SPECIFIC FUNCTIONS

def forward_grad_gpt2(model, weights, batch, criterion,
        metric, params):
    device = params['device']
    model.to(device)
    model.train()
    weights = weights.to(device)
    set_param_vec(model, weights)
    batch = tuple(input_tensor.to(device) for input_tensor in batch)
    mega_batch = batch
    grad_accum_steps = params['grad_accum_steps']
    batch_size = params['batch_size']
    train_batch_size = params['train_batch_size']
    #print(f"Real batch size: {mega_batch[3].size()[0]} from {train_batch_size} with {[b.size() for b in batch]}")
    train_batch_size = mega_batch[3].size()[0]
    accum_loss = None
    n_steps = train_batch_size // batch_size
    for i in range(n_steps):
        start = i * batch_size
        end = (i+1) * batch_size
        batch = [b[start:end] for b in mega_batch]
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        (lm_loss), (mc_loss), *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, lm_labels=lm_labels
        )
        loss = (lm_loss * params['lm_coef'] + mc_loss * params['mc_coef']) / n_steps
        if params["mean_grads"]:
            loss = loss / params["n_workers"]
        #print(f"Loss: {loss} from {lm_loss} and {mc_loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_norm'])
        if accum_loss is not None:
            accum_loss += loss
        else:
            accum_loss = loss
        #print(f"accum loss: {accum_loss} from {loss}")
    #print(f"accum loss: {accum_loss}")
    grad = get_grad(model, weights, params, device=device)
    # compress the gradient if needed
    if params["mode"] == "sketch":
        sketch = CSVec(d=params['grad_size'], c=params['num_cols'],
            r=params['num_rows'], device=device,
            numBlocks=params['num_blocks'])
        sketch.accumulateVec(grad)
        g = sketch.table.cpu()
    elif params["mode"] == "true_topk":
        g = grad
    elif params["mode"] == "local_topk":
        g = _topk(grad, k=params["k"])
    if accum_loss is not None:
        loss = accum_loss.item()/max(n_steps, 1)
    else:
        loss = 0
    """
    """
    return g, [loss]

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
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        model_outputs = model(input_ids, mc_token_ids, token_type_ids=token_type_ids)
        lm_logits, mc_logits, *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
        )
        lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
        return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
