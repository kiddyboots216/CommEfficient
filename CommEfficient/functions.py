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

from line_profiler import LineProfiler
import atexit
profile = LineProfiler()
#atexit.register(profile.print_stats)

g_worker_errors_sm = None
g_worker_velocities_sm = None
g_worker_transmitted_sm = None
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
        global g_worker_errors_sm
        global g_worker_velocities_sm
        global g_worker_transmitted_sm

        # ps_weights needs to be in shared memory so the workers can
        # update themselves with (possibly an approximation of) the
        # latest PS weights
        shape = (args.grad_size,)
        numel = int(np.prod(shape))
        g_ps_weights_sm = Array('f', numel, lock=False)
        ps_weights = sm2np(g_ps_weights_sm, shape)
        # store the initial weights of the model
        ps_weights[:] = param_vec[:]

        if args.do_topk_down:
            # client weights emulates each client's possibly stale weights
            shape = (num_clients, args.grad_size)
            numel = int(np.prod(shape))
            g_client_weights_sm = Array('f', numel, lock=False)
            # copy ps_weights into every row of client_weights
            client_weights = sm2np(g_client_weights_sm, shape)
            client_weights[:] = np.tile(param_vec, (num_clients, 1))
        else:
            g_client_weights_sm = Array('f', numel, lock=False)

        # errors and velocities hold the local error accumulation
        # vectors and local velocity vectors
        # transmitted holds what the workers sent to the PS
        shape = None
        if args.mode == "sketch":
            shape = (args.num_workers, args.num_rows, args.num_cols)
        elif args.mode in ["local_topk", "true_topk", "localSGD"]:
            shape = (args.num_workers, args.grad_size)
        numel = int(np.prod(shape))
        g_worker_errors_sm = Array('f', numel, lock=False)
        g_worker_velocities_sm = Array('f', numel, lock=False)
        g_worker_transmitted_sm = Array('f', numel, lock=False)

        # and zero them out
        e = sm2np(g_worker_errors_sm, shape)
        v = sm2np(g_worker_velocities_sm, shape)
        t = sm2np(g_worker_transmitted_sm, shape)
        e[:] = 0
        v[:] = 0
        t[:] = 0

        if args.share_ps_gpu:
            n_worker_gpus = args.num_devices
        else:
            n_worker_gpus = args.num_devices - 1
        # process pool that parallelizes training
        self.process_pool = multiprocessing.Pool(
                n_worker_gpus,
                initializer=worker.init_pool,
                initargs=(self.model, device, n_worker_gpus,
                          g_worker_errors_sm, g_worker_velocities_sm,
                          g_worker_transmitted_sm,
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

        # create momentum & error sketches -- one or one for each
        # client depending on whether we're doing virtual momentum
        if args.mode == "sketch":
            shape = (args.num_rows, args.num_cols)
        elif args.mode in ["true_topk", "local_topk", "localSGD"]:
            shape = (args.grad_size,)

        device = args.device
        self.Vvelocity = torch.zeros(shape).to(device)
        self.Verror = torch.zeros(shape).to(device)

    def get_lr(self):
        return get_lr(self.param_groups)

    @profile
    def step(self, client_indices):
        global g_ps_weights_sm
        global g_worker_transmitted_sm

        lr = self.get_lr()

        shape = None
        if self.args.mode == "sketch":
            shape = (self.args.num_workers,
                     self.args.num_rows, self.args.num_cols)
        elif self.args.mode in ["local_topk", "true_topk", "localSGD"]:
            shape = (self.args.num_workers, self.args.grad_size)
        transmitted = torch.from_numpy(sm2np(g_worker_transmitted_sm,
                                             shape))
        # if we're at the end of an epoch, the mini-batch, and therefore
        # the number of gradients, may be smaller than usual
        transmitted = transmitted[:len(client_indices)]
        transmitted = transmitted.to(self.args.device)

        weight_update, new_Vvelocity, new_Verror = get_server_update(
                transmitted,
                self.Vvelocity,
                self.Verror,
                self.args,
                lr)

        weight_update = weight_update.cpu()

        # a bit of a hack, but we also need to do momentum factor masking
        # on the worker momentum vectors for true_topk
        # which we can't do in the worker because we don't know the
        # global topk yet
        if self.args.mode == "true_topk":
            global g_worker_velocities_sm
            shape = (self.args.num_workers, self.args.grad_size)
            worker_velocities = sm2np(g_worker_velocities_sm, shape)
            worker_velocities = torch.from_numpy(worker_velocities)
            worker_velocities[torch.arange(len(client_indices)).view(-1,1),
                              weight_update.nonzero()[:,0]].zero_()

        # update ps_weights, momentums, and errors
        ps_weights = sm2np(g_ps_weights_sm, (self.args.grad_size,))
        ps_weights -= weight_update.numpy()

        self.Vvelocity[:] = new_Vvelocity
        self.Verror[:] = new_Verror

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

@profile
def get_server_update(transmitted, Vvelocity, Verror, args, lr):
    grads = None
    if args.mode in ["local_topk", "true_topk", "local_sgd"]:
        shape = (args.num_workers, args.grad_size)
    elif args.mode == "sketch":
        shape = (args.num_workers, args.num_rows, args.num_cols)

    helper = {"sketch": _server_helper_sketched,
              "local_topk": _server_helper_local_topk,
              "true_topk": _server_helper_true_topk,
              "localSGD": _server_helper_localSGD
             }[args.mode]

    weight_update, new_Vvelocity, new_Verror = helper(
            transmitted, Vvelocity, Verror, args, lr
        )

    return weight_update, new_Vvelocity, new_Verror

def agg_grads(grads, args):
    # aggregate the gradients
    if args.grad_reduction == "mean":
        # faster or about the same speed to sum on CPU, and no worries
        # about running out of memory
        if isinstance(grads, torch.sparse.FloatTensor):
            s = torch.sparse.sum
        else:
            s = torch.sum
        grad_agg = s(grads, dim=[0]) / args.num_workers
    if args.grad_reduction == "median":
        # numpy median is way faster than torch median
        grad_agg = torch.from_numpy(np.median(grads.cpu().numpy(), axis=0))

    return grad_agg.to(grads.device)

def _server_helper_localSGD(transmitted, Vvelocity, Verror, args, lr):
    update = agg_grads(transmitted, args)
    return update, Vvelocity, Verror

def _server_helper_true_topk(transmitted, Vvelocity, Verror, args, lr):
    assert args.error_type == "virtual"

    rho = args.virtual_momentum
    # Vvelocity = rho * Vvelocity + agg_grads(transmitted)
    torch.add(agg_grads(transmitted, args).to(args.device),
              Vvelocity,
              alpha=rho,
              out=Vvelocity)
    Verror += Vvelocity

    update = _topk(Verror, k=args.k)

    # error feedback
    Verror[update.nonzero()] = 0
    
    # momentum factor masking
    Vvelocity[update.nonzero()] = 0

    return update * lr, Vvelocity, Verror

@profile
def _server_helper_local_topk(transmitted, Vvelocity, Verror, args, lr):
    assert args.error_type == "local"

    """
    # make a sparse tensor of the local topk
    I = torch.meshgrid(torch.arange(grads.size()[0]),
                       torch.arange(args.k))[0]
    I = torch.stack((I, topk_i)).view(2, -1)
    V = topk_v.view(-1)
    size = (grads.size()[0], args.grad_size)
    topk_grads = torch.sparse_coo_tensor(I, V, size).to_dense()
    """

    rho = args.virtual_momentum
    agg = agg_grads(transmitted, args)
    torch.add(agg, Vvelocity, alpha=rho, out=Vvelocity)
    # ignore Verror, since you can't do virtual error with local topk

    # and no need to do momentum factor masking for the virtual
    # momentum, since that would zero out the entire Vvelocity every iter

    update = Vvelocity

    return update * lr, Vvelocity, Verror

def _server_helper_sketched(transmitted, Vvelocity, Verror, args, lr):
    rho = args.virtual_momentum
    k = args.k

    # must do the same type of momentum as error accumulation
    if args.error_type == "local":
        assert args.virtual_momentum == 0
    elif args.error_type == "virtual":
        assert args.local_momentum == 0

    agg = agg_grads(transmitted, args)
    torch.add(agg, Vvelocity, alpha=rho, out=Vvelocity)
    if args.error_type == "local":
        Verror = Vvelocity
    elif args.error_type == "virtual":
        Verror += Vvelocity

    sketch = args2sketch(args)
    sketch.accumulateTable(Verror)
    update = sketch.unSketch(k=args.k)

    # do virtual error
    sketch.zero()
    sketch.accumulateVec(update)
    sketched_update = sketch.table
    if args.error_type == "virtual":
        # this should work but doesn't (model diverges)
        #Verror -= sketched_update
        # instead, zero out Verror with sketched_update.nonzero()
        nz = sketched_update.nonzero()
        Verror[nz[:,0],nz[:,1]] = 0

    # momentum factor masking is annoying for sketched
    # to do it properly, we'd have to:
    # first, pull out the values of momentums where update is nonzero
    # then, sketch those values, and subtract them from momentums
    # this requires an unsketch of all num_workers momentum sketch,
    # which is expensive. So instead, just zero out the momentum sketch
    # anywhere where update is nonzero
    nz = sketched_update.nonzero()
    Vvelocity[nz[:,0], nz[:,1]].zero_()

    return update * lr, Vvelocity, Verror

def get_lr(optimizer_param_groups):
    if len(optimizer_param_groups) == 1:
        lr = optimizer_param_groups[0]["lr"]
        #print(f"Lr is {lr}")
        return lr

def split_results(results, n_results):
    return [np.array([r[i] for r in results]) for i in range(n_results)]
