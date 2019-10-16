import torch
from datetime import datetime
import os
import numpy as np
from csvec import CSVec
import copy
import time
import math

import cProfile

import ctypes
import multiprocessing
from multiprocessing import Array

import functions_worker as worker
from utils import sm2np, get_param_vec, set_param_vec, get_grad, _topk

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

def profile_helper(*args):
    cProfile.runctx("worker.update_forward_grad(*args)",
                    globals(), locals(),
                    "profile/init{:d}.prof".format(
                        multiprocessing.current_process()._identity[0]
                    )
                   )

class FedCommEffModel:
    def __init__(self, input_model, args):
        num_clients = args.num_clients
        participation = args.participation
        device = args.device
        cpu = "cpu"
        self.model = input_model
        param_vec = get_param_vec(self.model, cpu).numpy()
        grad_size = 0
        for p in self.model.parameters():
            if p.requires_grad:
                grad_size += torch.numel(p)
        args.grad_size = grad_size
        self.args = args

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
                                    param_vec.size * num_clients,
                                    lock=False)

        # copy ps_weights into every row of client_weights
        client_weights = sm2np(g_client_weights_sm,
                               (num_clients, param_vec.size))
        client_weights[:] = np.tile(param_vec, (num_clients, 1))

        # this shared memory block will hold the gradients
        # (or gradient sketches) computed by each worker in a round
        if args.mode == "sketch":
            g_worker_Sgrads_sm = Array(
                    'f',
                    args.num_workers * args.num_rows * args.num_cols,
                    lock=False
                )
            # and zero out worker_Sgrads to start
            worker_Sgrads_shape = (args.num_workers,
                                   args.num_rows,
                                   args.num_cols)
            worker_Sgrads = sm2np(g_worker_Sgrads_sm, worker_Sgrads_shape)
            worker_Sgrads[:] = 0
        elif args.mode in ["true_topk", "local_topk", "localSGD"]:
            g_worker_grads_sm = Array(
                    'f',
                    args.num_workers * args.grad_size,
                    lock=False
                )
            worker_grads_shape = (args.num_workers, args.grad_size)
            worker_grads = sm2np(g_worker_grads_sm, worker_grads_shape)
            worker_grads[:] = 0

        # process pool that parallelizes training
        self.process_pool = multiprocessing.Pool(
                args.num_workers,
                initializer=worker.init_pool,
                initargs=(self.model, device, args.num_devices-1,
                          g_worker_Sgrads_sm, g_worker_grads_sm,
                          g_client_weights_sm, g_ps_weights_sm)
            )


    def __del__(self):
        self.process_pool.close()
        self.process_pool.join()

    def train(self, training):
        self.training = training
    def save_pretrained(self, log_dir):
        global g_ps_weights_sm
        ps_weights = sm2np(g_ps_weights_sm,
                           (self.args.grad_size,))
        curr_weights = torch.from_numpy(ps_weights).cuda()
        set_param_vec(self.model, curr_weights)
        self.model.save_pretrained(log_dir)

    def __call__(self, batches, indices):
        global g_criterion
        global g_metric
        #global lr
        args = self.args
        if self.training:
            #self.args.lr = lr
            all_indices = []
            n_rows = math.ceil(len(indices)/args.num_devices)
            for n_row in range(n_rows):
                all_indices.append([])
                for num_device in range(args.num_devices):
                    num = n_row * args.num_devices + num_device
                    if num < len(indices):
                        all_indices[n_row].append(indices[num])
            all_results = []
            for gpu_indices in all_indices:
                args_tuples = [(i, idx,
                                batches[i], self.args,
                                g_criterion, g_metric)
                               for i, idx in enumerate(gpu_indices)]
                results = self.process_pool.starmap(
                        #profile_helper,
                        worker.update_forward_grad,
                        args_tuples
                    )
                all_results.extend(results)
            return split_results(results, self.args.num_results_train)

        else:
            args_tuples = [(batches[i], self.args, g_criterion, g_metric)
                           for i, idx in enumerate(indices)]
            results = self.process_pool.starmap(
                            worker.forward_multiprocessed,
                            args_tuples
                        )
            return split_results(results, self.args.num_results_val)

    def __getattr__(self, name):
        if name == "parameters":
            global g_ps_weights_sm

            ps_weights = sm2np(g_ps_weights_sm,
                               (self.args.grad_size,))
            curr_weights = torch.from_numpy(ps_weights)
            set_param_vec(self.model, curr_weights)
            return getattr(self.model, name)

    def zero_grad(self):
        self.process_pool.starmap(worker.zero_grad,
                              [() for _ in range(self.args.num_workers)])
        self.model.zero_grad()

class FedCommEffOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, args):
        #global grad_size
        grad_size = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    grad_size += torch.numel(p)
        self.args = args
        self.param_groups = optimizer.param_groups
        device = args.device

        # helper for rest of __init__
        def initialize_helper(thing_type, base_thing):
            if thing_type == "virtual":
                return [copy.deepcopy(base_thing)]
            elif thing_type == "local":
                return [copy.deepcopy(base_thing)
                        for _ in range(args.num_clients)]
            elif thing_type == "none":
                return None
            else:
                msg = "{} is an invalid type"
                raise ValueError(msg.format(thing_type))

        # hack so there isn't a not defined error later...
        self.base_sketch = None
        if args.mode == "sketch":
            sketch = CSVec(d=grad_size, c=args.num_cols,
                           r=args.num_rows, device=device,
                           numBlocks=args.num_blocks)
            # base_sketch can be used for miscellaneous
            # sketching activities
            self.base_sketch = sketch

            # create momentum & error sketches -- one or one for each
            # client depending on whether we're doing virtual momentum
            self.momentums = initialize_helper(
                    args.momentum_type, sketch
                )
            self.errors = initialize_helper(
                    args.error_type, sketch
                )
        elif args.mode in ["true_topk", "local_topk"]:
            # same as above but with vectors instead of sketches
            zero_vec = torch.zeros(args.grad_size).to(device)
            self.momentums = initialize_helper(
                    args.momentum_type, zero_vec
                )
            self.errors = initialize_helper(
                    args.error_type, zero_vec
                )
        elif args.mode == "localSGD":
            self.momentums = initialize_helper("none", None)
            self.errors = initialize_helper("none", None)

    def get_lr(self):
        new_lr = get_lr(self.param_groups)
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
        if self.args.momentum_type == "virtual":
            cur_momentums = self.momentums
        elif self.args.momentum_type == "local":
            cur_momentums = [self.momentums[idx] for idx in indices]
        elif self.args.momentum_type == "none":
            cur_momentums = None
        else:
            msg = "invalid momentum type {}"
            raise ValueError(msg.format(self.args.momentum_type))

        cur_errors = None
        if self.args.error_type == "virtual":
            cur_errors = self.errors
        elif self.args.error_type == "local":
            cur_errors = [self.errors[idx] for idx in indices]
        elif self.args.error_type == "none":
            cur_errors = None
        else:
            msg = "invalid error type {}"
            raise ValueError(msg.format(self.args.error_type))

        new_ps_weights, new_momentums, new_errors = get_updated_server(
                cur_momentums,
                cur_errors,
                self.args,
                lr,
                self.base_sketch)

        # update ps_weights, momentums, and errors
        ps_weights = sm2np(g_ps_weights_sm, (self.args.grad_size,))
        ps_weights[:] = new_ps_weights

        if self.args.momentum_type == "virtual":
            self.momentums = new_momentums
        elif self.args.momentum_type == "local":
            for i, idx in enumerate(indices):
                self.momentums[idx] = new_momentums[i]

        if self.args.error_type == "virtual":
            self.errors = new_errors
        elif self.args.error_type == "local":
            for i, idx in enumerate(indices):
                self.errors[idx] = new_errors[i]

    def zero_grad(self):
        pass

class FedCommEffCriterion:
    def __init__(self, input_criterion, args):
        global g_criterion
        g_criterion = input_criterion
    def __call__(self, *args):
        global g_criterion
        out = g_criterion(*args)
        return out
class FedCommEffMetric:
    def __init__(self, input_metric, args):
        global g_metric
        g_metric = input_metric
    def __call__(self, *args):
        global g_metric
        out = g_metric(*args)
        return out

def get_updated_server(momentums, errors, args, lr, sketch=None):
    if args.mode == "sketch":
        return _server_helper_sketched(momentums, errors, args,
                                       lr, sketch)
    elif args.mode == "local_topk":
        return _server_helper_local_topk(momentums, errors, args, lr)
    elif args.mode == "true_topk":
        return _server_helper_true_topk(momentums, errors, args, lr)
    elif args.mode == "localSGD":
        return _server_helper_localSGD(momentums, errors, args, lr)
    else:
        assert False, "invalid mode {}".format(args.mode)

def _server_helper_localSGD(momentum_vecs, error_vecs, args, lr):
    global ps_weights_sm
    global p_worker_grads_sm
    device = torch.device(args.device)
    ps_weights = sm2np(g_ps_weights_sm, (args.grad_size,),)
    ps_weights = torch.from_numpy(ps_weights).to(device)
    worker_grads_shape = (args.num_workers, args.grad_size)
    worker_grads = sm2np(g_worker_grads_sm, worker_grads_shape)
    large = args.model == "gpt2"
    if large:
        grad_sum = torch.from_numpy(worker_grads[0]).to(device)
        for g in worker_grads[1:]:
            grad_sum += torch.from_numpy(g).to(device)
    else:
        grad_sum = np.sum([torch.from_numpy(g).to(device) for g in worker_grads])
    update = grad_sum
    return (ps_weights - update).cpu(), momentum_vecs, error_vecs

def _server_helper_true_topk(momentum_vecs, error_vecs, args, lr):
    global g_ps_weights_sm
    global g_worker_grads_sm
    assert args.momentum_type == "virtual"
    assert args.error_type == "virtual"

    device = torch.device(args.device)
    momentum = args.momentum

    ps_weights = sm2np(g_ps_weights_sm, (args.grad_size,),)
    ps_weights = torch.from_numpy(ps_weights).to(device)

    worker_grads_shape = (args.num_workers, args.grad_size)
    worker_grads = sm2np(g_worker_grads_sm, worker_grads_shape)
    large = args.model == "gpt2"
    if args.grad_reduction == "mean":
        if large:
            grad_sum = torch.from_numpy(worker_grads[0]).to(device)
            for g in worker_grads[1:]:
                grad_sum += torch.from_numpy(g).to(device)
            grad_sum /= args.num_workers
        else:
            grad_sum = np.sum([torch.from_numpy(g).to(device) for g in worker_grads])
            grad_sum /= args.num_workers
    if args.grad_reduction == "median":
        grad_agg = np.median([torch.from_numpy(g).to(device) for g in worker_grads])
    momentum_vec = momentum_vecs[0]
    error_vec = error_vecs[0]
    momentum_vec *= momentum
    momentum_vec += grad_sum
    error_vec += momentum_vec
    update = _topk(error_vec, k=args.k)
    momentum_vec[update.nonzero()] = 0
    error_vec[update.nonzero()] = 0
    #print(f"Updating {ps_weights.mean()} with {update.mean()} * {lr}")
    return (ps_weights - update * lr).cpu(), [momentum_vec], [error_vec]

def _server_helper_local_topk(momentum_vecs, error_vecs, args, lr):
    global g_ps_weights_sm
    global g_worker_grads_sm
    assert args.momentum_type == "virtual"
    assert args.error_type == "virtual"

    device = torch.device(args.device)
    momentum = args.momentum

    ps_weights = sm2np(g_ps_weights_sm, (args.grad_size,),)
    ps_weights = torch.from_numpy(ps_weights).to(device)

    worker_grads_shape = (args.num_workers, args.grad_size)
    worker_grads = sm2np(g_worker_grads_sm, worker_grads_shape)
    large = args.model == "gpt2"
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
    update = _topk(error_vec, k=args.k)
    momentum_vec[update.nonzero()] = 0
    error_vec[update.nonzero()] = 0
    return (ps_weights - update * lr).cpu(), [momentum_vec], [error_vec]

def _server_helper_sketched(momentum_sketches, error_sketches,
                            args, lr, sketch):
    momentum = args.momentum
    k = args.k
    device = torch.device(args.device)
    momentum_type = args.momentum_type
    error_type = args.error_type
    local = momentum_type == 'local' and error_type == 'local'
    virtual = momentum_type == 'virtual' and error_type == 'virtual'
    none = momentum_type == 'none' and error_type == 'none'
    assert local or virtual or none

    global g_ps_weights_sm
    global g_worker_Sgrads_sm

    ps_weights = sm2np(g_ps_weights_sm, (args.grad_size,),)
    ps_weights = torch.from_numpy(ps_weights).to(device)

    worker_Sgrads_shape = (args.num_workers,
                           args.num_rows,
                           args.num_cols)
    worker_Sgrads = sm2np(g_worker_Sgrads_sm, worker_Sgrads_shape)
    worker_Sgrads = [torch.from_numpy(Sg).to(device)
                     for Sg in worker_Sgrads]
    if local:
        for grad_table, momentum_sketch, error_sketch in zip(worker_Sgrads, momentum_sketches, error_sketches):
            if args.momentum_type != "none":
                momentum_sketch *= momentum
                momentum_sketch.accumulateTable(grad_table)
                if args.error_type != "none":
                    error_sketch += momentum_sketch
            elif args.error_type != "none":
                error_sketch.accumulateTable(grad_table)
            else:
                sketch += grad_sketch
        if args.error_type != "none":
            update = np.sum(error_sketches).unSketch(k=k)
        else:
            update = sketch.unSketch(k=k)
        sketch.zero()
        sketch.accumulateVec(update)
        hh_coords = sketch.table.nonzero()
        hh_0, hh_1 = hh_coords[:, 0], hh_coords[:, 1]
        for momentum_sketch, error_sketch in zip(momentum_sketches, error_sketches):
            if args.momentum_type != "none":
                momentum_sketch.table[hh_0, hh_1] = 0
            if args.error_type != "none":
                error_sketch.table[hh_0, hh_1] = 0

    elif virtual:
        if args.grad_reduction == "mean":
            grad_sketch_agg = sketch
            grad_sketch_agg.zero()
            for S in worker_Sgrads:
                grad_sketch_agg.accumulateTable(S)
            grad_sketch_agg /= args.num_workers
        elif args.grad_reduction == "median":
            sketch.zero()
            csvecs = [copy.deepcopy(sketch) for _ in worker_Sgrads]
            for csvec, S in zip(csvecs, worker_Sgrads):
                csvec.accumulateTable(S)
            grad_sketch_agg = csvec.median(csvecs)

        momentum_sketch = momentum_sketches[0]
        error_sketch = error_sketches[0]
        if args.momentum_type != "none":
            momentum_sketch *= momentum
            momentum_sketch += grad_sketch_agg
            if args.error_type != "none":
                error_sketch += momentum_sketch
        elif args.error_type != "none":
            error_sketch += grad_sketch_agg
        else:
            sketch += grad_sketch_agg
        if args.error_type != "none":
            update = error_sketch.unSketch(k=k)
        elif args.momentum_type != "none":
            update = momentum_sketch.unSketch(k=k)
        else:
            update = sketch.unSketch(k=k)
        sketch.zero()
        sketch.accumulateVec(update)
        hh_coords = sketch.table.nonzero()
        hh_0, hh_1 = hh_coords[:, 0], hh_coords[:, 1]
        if args.momentum_type != "none":
            momentum_sketch.table[hh_0, hh_1] = 0
        if args.error_type != "none":
            error_sketch.table[hh_0, hh_1] = 0
        momentum_sketches = [momentum_sketch]
        error_sketches = [error_sketch]

    elif none:
        global g_worker_grads_sm
        worker_grads_shape = (args.num_workers, args.grad_size)
        worker_grads = sm2np(g_worker_grads_sm, worker_grads_shape)
        grad_sum = np.sum(worker_grads)
        grad_sketch_agg = sketch
        grad_sketch_agg.zero()
        for S in worker_Sgrads:
            grad_sketch_agg.accumulateTable(S)
        update = grad_sketch_agg.unSketch(k=k)
        print(f"Reconstruction error: {(update - grad_sum).norm()}")
    return (ps_weights - update * lr).cpu(), momentum_sketches, error_sketches

def get_lr(optimizer_param_groups):
    if len(optimizer_param_groups) == 1:
        lr = optimizer_param_groups[0]["lr"]
        #print(f"Lr is {lr}")
        return lr

def split_results(results, n_results):
    return [np.array([r[i] for r in results]) for i in range(n_results)]
