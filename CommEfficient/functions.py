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

        if args.share_ps_gpu:
            n_worker_gpus = args.num_devices
        else:
            n_worker_gpus = args.num_devices - 1
        # process pool that parallelizes training
        self.process_pool = multiprocessing.Pool(
                n_worker_gpus,
                initializer=worker.init_pool,
                initargs=(self.model, device, n_worker_gpus,
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
        curr_weights = torch.from_numpy(ps_weights)
        set_param_vec(self.model, curr_weights)
        self.model.save_pretrained(log_dir)

    def __call__(self, batches, indices):
        global g_criterion
        global g_metric
        args = self.args
        if self.training:
            #self.args.lr = lr
            args_tuples = [(i, idx,
                            batches[i], self.args,
                            g_criterion, g_metric)
                           for i, idx in enumerate(indices)]

            results = self.process_pool.starmap(
                    #profile_helper,
                    worker.update_forward_grad,
                    args_tuples
                )
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
        # use the last GPU for the PS
        if args.device[:4] == "cuda":
            torch.cuda.set_device(args.num_devices-1)
        device = args.device

        # this was probably already calculated in FedCommEffModel,
        # but just in case not, we recompute it here
        grad_size = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    grad_size += torch.numel(p)
        self.args = args
        args.grad_size = grad_size

        self.param_groups = optimizer.param_groups

        # helper for rest of __init__
        def type2N(thing_type, num_clients):
            if thing_type == "virtual":
                return 1
            elif thing_type == "local":
                return num_clients
            elif thing_type == "none":
                return 0
            else:
                msg = "{} is an invalid type"
                raise ValueError(msg.format(thing_type))

        # create momentum & error sketches -- one or one for each
        # client depending on whether we're doing virtual momentum
        if args.mode == "sketch":
            shape = (args.num_rows, args.num_cols)
        elif args.mode in ["true_topk", "local_topk", "localSGD"]:
            shape = (args.grad_size,)

        momentumN = type2N(args.momentum_type, args.num_clients)
        errorN = type2N(args.error_type, args.num_clients)

        self.momentums = torch.zeros((momentumN,) + shape)
        self.errors = torch.zeros((errorN,) + shape)

    def get_lr(self):
        return get_lr(self.param_groups)

    def step(self, client_indices, ret=False):
        # in this method we're agnostic as to whether we're sketched,
        # true topk, or local topk
        lr = self.get_lr()

        if self.args.momentum_type == "virtual":
            momentum_indices = [0]
        elif self.args.momentum_type == "local":
            momentum_indices = client_indices
        elif self.args.momentum_type == "none":
            momentum_indices = ()
        else:
            msg = "invalid momentum type {}"
            raise ValueError(msg.format(self.args.momentum_type))

        cur_momentums = self.momentums[momentum_indices]

        if self.args.error_type == "virtual":
            error_indices = [0]
        elif self.args.error_type == "local":
            error_indices = client_indices
        elif self.args.error_type == "none":
            error_indices = ()
        else:
            msg = "invalid error type {}"
            raise ValueError(msg.format(self.args.error_type))

        cur_errors = self.errors[error_indices]

        new_ps_weights, new_momentums, new_errors = get_updated_server(
                cur_momentums,
                cur_errors,
                self.args,
                lr)

        # update ps_weights, momentums, and errors
        ps_weights = sm2np(g_ps_weights_sm, (self.args.grad_size,))
        ps_weights[:] = new_ps_weights.cpu().numpy()

        self.momentums[momentum_indices] = new_momentums.cpu()
        self.errors[error_indices] = new_errors.cpu()

    def zero_grad(self):
        raise NotImplementedError("Please call zero_grad() on the model instead")

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

def args2sketch(args):
    return CSVec(d=args.grad_size, c=args.num_cols,
                 r=args.num_rows, device=args.device,
                 numBlocks=args.num_blocks)

def get_updated_server(momentums, errors, args, lr):
    global g_worker_grads_sm
    global g_worker_Sgrads_sm

    grads = None
    if args.mode in ["true_topk", "local_topk", "local_sgd"]:
        worker_grads_shape = (args.num_workers, args.grad_size)
        grads = sm2np(g_worker_grads_sm, worker_grads_shape)
    elif args.mode == "sketch":
        worker_Sgrads_shape = (args.num_workers,
                               args.num_rows, args.num_cols)
        grads = sm2np(g_worker_Sgrads_sm, worker_Sgrads_shape)

    # doing arithmetic on tensors appears to be ~2x faster in torch
    # than in numpy for some reason
    grads = torch.from_numpy(grads)

    # if we're at the end of an epoch, the mini-batch, and therefore
    # the number of gradients, may be smaller than usual
    max_worker_id = max(momentums.size()[0], errors.size()[0])
    grads = grads[:max_worker_id]

    if args.mode == "sketch":
        u = _server_helper_sketched(grads, momentums, errors, args, lr)
    elif args.mode == "local_topk":
        u = _server_helper_local_topk(grads, momentums, errors, args, lr)
    elif args.mode == "true_topk":
        u = _server_helper_true_topk(grads, momentums, errors, args, lr)
    elif args.mode == "localSGD":
        u = _server_helper_localSGD(grads, momentums, errors, args, lr)
    else:
        assert False, "invalid mode {}".format(args.mode)

    weight_update, new_momentums, new_errors = u

    ps_weights = sm2np(g_ps_weights_sm, (args.grad_size,))
    ps_weights = torch.from_numpy(ps_weights)

    return ps_weights - weight_update.cpu(), new_momentums.cpu(), new_errors.cpu()

def agg_grads(grads, args):
    # aggregate the gradients
    if args.grad_reduction == "mean":
        # faster or about the same speed to sum on CPU, and no worries
        # about running out of memory
        grad_agg = torch.sum(grads, dim=0) / args.num_workers
    if args.grad_reduction == "median":
        # numpy median is way faster than torch median
        grad_agg = torch.from_numpy(np.median(grads.cpu().numpy(), axis=0))

    return grad_agg.to(grads.device)

def _server_helper_localSGD(grads, momentums, errors, args, lr):
    update = agg_grads(grads, args)
    return update, momentums, errors

def _server_helper_true_topk(grads, momentum, error, args, lr):
    assert args.momentum_type == "virtual"
    assert args.error_type == "virtual"

    rho = args.momentum
    momentum = rho * momentum + agg_grads(grads, args).to(args.device)
    error += momentum

    update = _topk(error, k=args.k)

    momentum[update.nonzero()] = 0
    error[update.nonzero()] = 0

    #print(f"Updating {ps_weights.mean()} with {update.mean()} * {lr}")
    return update * lr, momentum, error

def _server_helper_local_topk(grads, momentums, errors, args, lr):
    assert args.error_type == "local"

    if args.momentum_type == "none":
        # calculate what the workers sent
        transmitted = torch.zeros_like(errors)
        errors += grads
        for i, error in enumerate(errors):
            transmitted[i,:] = _topk(error.to(args.device), k=args.k).cpu()

            # zero out locally what was transmitted
            # need to do this one worker at a time to not OOM
            nz = transmitted[i,:].nonzero()
            errors[i, nz].zero_()

        update = agg_grads(transmitted, args)
    elif args.momentum_type == "local":
        rho = args.momentum
        torch.add(grads, momentums, alpha=rho, out=momentums)

        transmitted = torch.zeros_like(errors)
        errors += momentums
        for i, error in enumerate(errors):
            transmitted[i,:] = _topk(error.to(args.device), k=args.k).cpu()

            # zero out locally what was transmitted
            nz = transmitted[i,:].nonzero()
            momentums[i, nz].zero_()
            errors[i, nz].zero_()

        update = agg_grads(transmitted, args)

    elif args.momentum_type == "virtual":
        # only one momentum vec if virtual
        momentum = momentums

        # first figure out what to transmit, then do momentum on the server
        errors += grads
        transmitted = torch.zeros_like(errors)
        for i, error in enumerate(errors):
            transmitted[i,:] = _topk(error.to(args.device), k=args.k).cpu()
            nz = transmitted[i,:].nonzero()
            errors[i, nz].zero_()

        rho = args.momentum
        torch.add(agg_grads(transmitted, args),
                  momentum,
                  alpha=rho,
                  out=momentum)
        update = momentum
        # no momentum factor masking! b/c it doesn't make sense in this case

        # call it momentums again for the return statement
        momentums = momentum


    return update * lr, momentums, errors

def _server_helper_sketched(Sgrads, Smomentums, Serrors, args, lr):
    rho = args.momentum
    k = args.k

    # do everything on the GPU
    Sgrads = Sgrads.to(args.device)
    Smomentums = Smomentums.to(args.device)
    Serrors = Serrors.to(args.device)

    momentum_type = args.momentum_type
    error_type = args.error_type
    local = momentum_type == 'local' and error_type == 'local'
    virtual = momentum_type == 'virtual' and error_type == 'virtual'
    none = momentum_type == 'none' and error_type == 'none'
    assert local or virtual or none

    sketch = args2sketch(args)
    if none:
        transmitted = Sgrads
        sketch.accumulateTable(agg_grads(Sgrads, args))
        update = sketch.unSketch(k=k)
    elif local:
        Smomentums = rho * Smomentums + Sgrads
        Serrors += Smomentums
        transmitted = Serrors

        sketch.accumulateTable(agg_grads(transmitted, args))
        Supdate = sketch.table
        # do error feedback in the sketch
        Serrors -= Supdate

        update = sketch.unSketch(k=k)

        # momentum factor masking is annoying for sketched
        # to do it properly, we'd have to:
        # first, pull out the values of momentums where update is nonzero
        # then, sketch those values, and subtract them from momentums
        # this requires an unsketch of all num_workers momentum sketch,
        # which is expensive. So instead, just zero out the momentum sketch
        # anywhere where update is nonzero
        nz = Supdate.nonzero()
        Smomentums[0, nz[:,0], nz[:,1]] = 0

    elif virtual:
        transmitted = Sgrads

        # for virtual, we only have one momentum and one error sketch
        Smomentum = Smomentums
        Serror = Serrors

        Smomentum = rho * Smomentum + agg_grads(transmitted, args)
        Serror += Smomentum
        # Serror is 1xrowsxcols, so need to index by 0
        sketch.accumulateTable(Serror[0])
        Supdate = sketch.table
        # error feedback within the sketch
        Serror -= Supdate

        update = sketch.unSketch(k=k)

        # see comment about momentum factor masking with sketching above
        nz = Supdate.nonzero()
        # (still 3-dimensional, but the first dimension has size 1)
        Smomentum[0, nz[:,0], nz[:,1]] = 0

        # rename to plurals for the return statement...
        Smomentums = Smomentum
        Serrors = Serror

    return update * lr, Smomentums, Serrors

def get_lr(optimizer_param_groups):
    if len(optimizer_param_groups) == 1:
        lr = optimizer_param_groups[0]["lr"]
        #print(f"Lr is {lr}")
        return lr

def split_results(results, n_results):
    return [np.array([r[i] for r in results]) for i in range(n_results)]
