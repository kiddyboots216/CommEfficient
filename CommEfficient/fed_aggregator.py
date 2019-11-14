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
import torch.multiprocessing as multiprocessing

import fed_worker as worker
from utils import get_param_vec, set_param_vec, get_grad, _topk

#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#world_size = comm.Get_size()

#from line_profiler import LineProfiler
#import atexit
#profile = LineProfiler()
#atexit.register(profile.print_stats)

g_client_errors = None
g_client_velocities = None
g_client_transmitted = None
g_client_weights = None
g_ps_weights = None

g_criterion = None
g_accuracy = None

# a bit hacky, but the model needs to tell the optimizer how many
# workers actually participated in a round (e.g. at the end of an epoch
# there might not be enough data left to have args.num_workers workers)
g_num_valid_workers = 0

def profile_helper(*args):
    cProfile.runctx("worker.update_forward_grad(*args)",
                    globals(), locals(),
                    "profile/cifar_ltk.{:d}.prof".format(
                        multiprocessing.current_process()._identity[0]
                    )
                   )

class FedModel:
    def __init__(self, input_model, args):
        num_clients = args.num_clients
        device = args.device
        cpu = "cpu"
        self.model = input_model
        param_vec = get_param_vec(self.model, cpu)
        grad_size = 0
        for p in self.model.parameters():
            if p.requires_grad:
                grad_size += torch.numel(p)
        args.grad_size = grad_size
        self.args = args

        global g_ps_weights
        global g_client_weights
        global g_client_errors
        global g_client_velocities
        global g_worker_transmitted

        # ps_weights needs to be in shared memory so the workers can
        # update themselves with (possibly an approximation of) the
        # latest PS weights
        g_ps_weights = torch.zeros(args.grad_size).share_memory_()
        # store the initial weights of the model
        g_ps_weights[:] = param_vec[:]

        # everyone gets ps_weights at the beginning of each round if
        # not args.topk_down, so no need for this array
        if args.do_topk_down:
            # client weights emulates each client's possibly stale weights
            shape = (num_clients, args.grad_size)
            g_client_weights = torch.zeros(shape).share_memory_()
            # copy ps_weights into every row of client_weights
            g_client_weights[:] = param_vec.repeat(num_clients, 1)

        # errors and velocities hold the local error accumulation
        # vectors and local velocity vectors
        # transmitted holds what the workers sent to the PS
        shape = None
        if args.mode == "sketch":
            shape = (args.num_clients, args.num_rows, args.num_cols)
        elif args.mode in ["local_topk", "true_topk", "localSGD"]:
            shape = (args.num_clients, args.grad_size)

        # don't make these arrays unless we need them
        if args.error_type == "local" or args.local_momentum > 0:
            g_client_errors = torch.zeros(shape).share_memory_()
            g_client_velocities = torch.zeros(shape).share_memory_()

        # there are only num_workers transmitted vectors
        shape = (args.num_workers,) + shape[1:]
        g_worker_transmitted = torch.zeros(shape).share_memory_()

        if args.share_ps_gpu:
            self.n_worker_gpus = args.num_devices
        else:
            self.n_worker_gpus = args.num_devices - 1
        # process pool that parallelizes training
        self.process_pool = multiprocessing.Pool(
                self.n_worker_gpus,
                initializer=worker.init_pool,
                initargs=(self.model, device, self.n_worker_gpus,
                          g_client_errors, g_client_velocities,
                          g_worker_transmitted,
                          g_client_weights, g_ps_weights)
            )
        self.hook = hook


    def finalize(self):
        self.process_pool.close()
        self.process_pool.join()

    def train(self, training):
        self.training = training
    def save_pretrained(self, log_dir):
        global g_ps_weights
        set_param_vec(self.model, g_ps_weights)
        self.model.save_pretrained(log_dir)

    def __call__(self, batch):
        global g_criterion
        global g_metric
        args = self.args

        # batch is a tuple, with the client ids as the first tensor
        client_indices = batch[0]
        batch = batch[1:]

        if self.training:

            unique_clients = torch.unique(client_indices)

            # this is to tell the optimizer how many workers actually
            # participated this round
            global g_num_valid_workers
            g_num_valid_workers = unique_clients.numel()

            worker_batches = [tuple(t[torch.where(client_indices == i)[0]]
                                    for t in batch)
                              for i in unique_clients]

            args_tuples = [(i, idx,
                            worker_batches[i], self.args,
                            g_criterion, g_metric, self.hook)
                           for i, idx in enumerate(unique_clients)]

            results = self.process_pool.starmap(
                    #profile_helper,
                    worker.update_forward_grad,
                    args_tuples
                )
            #return [(1,2,3,4),(5,6,7,8)]
            return split_results(results, self.args.num_results_train)

        else:
            split = [t.split(args.local_batch_size) for t in batch]
            num_shards = len(split[0])
            batch_shards = [tuple(l[i] for l in split)
                            for i in range(num_shards)]
            args_tuples = [(batch_shard, self.args,
                            g_criterion, g_metric)
                           for batch_shard in batch_shards]
            results = self.process_pool.starmap(
                            worker.forward_multiprocessed,
                            args_tuples
                        )
            return split_results(results, self.args.num_results_val)

    def __getattr__(self, name):
        if name == "parameters":
            global g_ps_weights
            set_param_vec(self.model, g_ps_weights)
            return getattr(self.model, name)

    def zero_grad(self):
        self.process_pool.starmap(worker.zero_grad,
                              [() for _ in range(self.args.num_workers)])
        self.model.zero_grad()

class FedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, args):
        # use the last GPU for the PS
        if args.device[:4] == "cuda":
            torch.cuda.set_device(args.num_devices-1)
        device = args.device

        # this was probably already calculated in FedModel,
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

    #@profile
    def step(self):
        global g_ps_weights
        global g_worker_transmitted
        global g_num_valid_workers

        lr = self.get_lr()

        # if we're at the end of an epoch, the mini-batch, and therefore
        # the number of gradients, may be smaller than usual
        transmitted = g_worker_transmitted[:g_num_valid_workers]
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
        if self.args.mode == "true_topk" and args.local_momentum > 0:
            global g_client_velocities
            rows = torch.arange(g_num_valid_workers).view(-1,1)
            g_client_velocities[rows, weight_update.nonzero()[:,0]].zero_()

        # update ps_weights, momentums, and errors
        g_ps_weights -= weight_update

        self.Vvelocity[:] = new_Vvelocity
        self.Verror[:] = new_Verror

    def zero_grad(self):
        raise NotImplementedError("Please call zero_grad() on the model instead")

class FedCriterion:
    def __init__(self, input_criterion, args):
        global g_criterion
        g_criterion = input_criterion
    def __call__(self, *args):
        global g_criterion
        out = g_criterion(*args)
        return out

class FedMetric:
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

#@profile
def get_server_update(transmitted, Vvelocity, Verror, args, lr):
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

#@profile
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
