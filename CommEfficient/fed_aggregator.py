import torch
from datetime import datetime
import os
import numpy as np
from csvec import CSVec
import copy
import time
import math
from collections import deque
import warnings

import cProfile

import ctypes
import torch.multiprocessing as multiprocessing

import fed_worker as worker
from utils import get_param_vec, set_param_vec, get_grad, _topk, clip_grad

# gets multipled by 1/participation. See use below
DEQUE_MAXLEN_MULT = 10

def shms():
    pid = os.getpid()
    return [s for s in os.listdir("/dev/shm") if str(pid) in s]

#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#world_size = comm.Get_size()

#from line_profiler import LineProfiler
#import atexit
#profile = LineProfiler()
#atexit.register(profile.print_stats)

g_ps_weights = None
g_minibatch_gradient = None
# need client velocities to be global so the optimizer can update them
# in true topk after computing the weight update
g_client_velocities = None
g_participating_clients = None
# need a global LR so FedModel can send it to workers when doing fedavg
g_lr = None

def profile_helper(*args):
    cProfile.runctx("worker.update_forward_grad(*args)",
                    globals(), locals(),
                    "profile/cifar_fedsampler.{:d}.prof".format(
                        multiprocessing.current_process()._identity[0]
                    )
                   )

class FedModel:
    def __init__(self, input_model, compute_loss, args,
                 compute_loss_val=None, compute_loss_mal=None):
        global g_minibatch_gradient
        global g_ps_weights
        global g_client_velocities
        global g_lr

        # use the last GPU for the PS
        if args.device[:4] == "cuda":
            torch.cuda.set_device(args.num_devices-1)

        num_clients = args.num_clients
        # TODO hack
        if args.num_clients is None:
            num_clients = {"EMNIST": 3500,
                           "CIFAR10": None, # don't support non-iid cifar
                           None: 175468 # personachat
                           }[args.dataset_name]
        self.num_clients = num_clients

        device = args.device
        self.model = input_model
        self.compute_loss_train = compute_loss
        self.compute_loss_val = (compute_loss_val
                                 if compute_loss_val is not None
                                 else compute_loss)
        self.compute_loss_mal = (compute_loss_mal
                                if compute_loss_mal is not None
                                else compute_loss)
        self.mal_id = args.mal_id
        param_vec = []
        grad_size = 0
        for p in self.model.parameters():
            if p.requires_grad:
                grad_size += torch.numel(p)
                param_vec.append(p.data.view(-1))
        param_vec = torch.cat(param_vec)
        args.grad_size = grad_size
        print("grad_size", args.grad_size)
        self.args = args

        # ps_weights needs to be in shared memory so the workers can
        # update themselves with (possibly an approximation of) the
        # latest PS weights
        g_ps_weights = torch.zeros(args.grad_size).to(args.device).float()
        # store the initial weights of the model
        g_ps_weights[:] = param_vec[:]

        if (self.args.num_epochs <= 1
                and self.args.local_batch_size == -1):
            # keeping track of download bytes is simpler in this
            # case (see comments in _call_train)
            self.updated_since_init = torch.zeros(args.grad_size,
                                                  dtype=torch.bool,
                                                  device=args.device)
            self.prev_ps_weights = g_ps_weights.clone()
        else:
            # keeping track of download bytes is harder (see comments
            # in _call_train)

            # keep a deque of past ps_weights so we can calculate
            # number of bytes a sparsely participating worker needs
            # to download each iteration

            # if clients are picked randomly, then a single client
            # getting picked is a poisson process with rate
            # participation, so the probability that a client won't
            # be picked for more than 10/participation iterations
            # is 1/e^10  < 0.005%
            participation = args.num_workers / num_clients
            maxlen = int(DEQUE_MAXLEN_MULT / participation)
            self.ps_weights_history = deque([], maxlen=maxlen)
            self.client_num_stale_iters = torch.zeros(num_clients).long()

        # g_lr is the current LR (used for fedavg) updated by
        # FedOptimizer
        g_lr = torch.zeros(1).float().share_memory_()

        # everyone gets ps_weights at the beginning of each round if
        # not args.topk_down, so no need for this array
        if args.do_topk_down:
            # client weights emulates each client's possibly stale
            # weights
            shape = (num_clients, args.grad_size)
            self.client_weights = torch.zeros(shape).share_memory_()
            # copy ps_weights into every row of client_weights
            self.client_weights[:] = param_vec.repeat(num_clients, 1)

        # errors and velocities hold the local error accumulation
        # vectors and local velocity vectors
        # transmitted holds what the workers sent to the PS
        shape = None
        if args.mode == "sketch":
            shape = (num_clients, args.num_rows, args.num_cols)
        elif args.mode in ["local_topk", "true_topk", "fedavg",
                           "uncompressed"]:
            shape = (num_clients, args.grad_size)
        print(f"Shared memory array shape: {shape}")

        # don't make these arrays unless we need them
        if args.error_type == "local" or args.local_momentum > 0:
            self.client_errors = torch.zeros(shape).share_memory_()
            g_client_velocities = torch.zeros(shape).share_memory_()

        g_minibatch_gradient = torch.zeros(shape[1:]).to(args.device)

        if args.share_ps_gpu:
            n_worker_gpus = args.num_devices
        else:
            n_worker_gpus = args.num_devices - 1

        # queues to send batches to worker processes and receive results
        self.batches_queues = [multiprocessing.Queue()
                               for _ in range(n_worker_gpus)]
        self.results_queues = [multiprocessing.Queue()
                               for _ in range(n_worker_gpus)]

        # start processes to run update_forward_grad
        self.update_forward_grad_ps = []
        world_size = n_worker_gpus + 1
        for i in range(n_worker_gpus):
            p = multiprocessing.Process(
                        target=worker.worker_loop,
                        args=(self.model, g_ps_weights,
                              self.client_weights,
                              self.client_errors, g_client_velocities,
                              self.batches_queues[i],
                              self.results_queues[i], g_lr,
                              i + 1, world_size,
                              self.compute_loss_train,
                              self.compute_loss_val, 
                              self.compute_loss_mal, args)
                    )
            p.start()
            self.update_forward_grad_ps.append(p)

        # set up communication channel with worker processes
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(args.port)
        torch.distributed.init_process_group("nccl", rank=0,
                                             world_size=world_size)

    def finalize(self):
        # tell workers we're done
        for q in self.batches_queues:
            q.put(None)
        # end workers
        for p in self.update_forward_grad_ps:
            p.join()
            p.close()

    def train(self, training):
        self.training = training

    def save_pretrained(self, log_dir):
        global g_ps_weights
        set_param_vec(self.model, g_ps_weights.cpu())
        self.model.save_pretrained(log_dir)

    def _call_train(self, batch):
        global g_minibatch_gradient
        global g_lr

        # batch is a tuple, with the client ids as the first tensor
        client_indices = batch[0]
        unique_clients = torch.unique(client_indices)
        g_participating_clients = unique_clients

        worker_batches = [tuple(t[torch.where(client_indices == i)[0]]
                                for t in batch)
                          for i in unique_clients]

        # assign worker batches to processes
        # each process should get ~an equal number of batches to compute
        # gradients on. The process will sum gradients locally, then
        # will reduce/sum across all processes
        per_proc = len(worker_batches) // len(self.update_forward_grad_ps)
        proc_batches = [worker_batches[i:i + per_proc]
                        for i in range(0, len(worker_batches), per_proc)]

        download_bytes = None
        if (self.args.num_epochs <= 1
                and self.args.local_batch_size == -1):
            # we can just maintain a single boolean tensor which is
            # True if the corresponding weight has been updated
            # since the beginning of training
            diff = g_ps_weights - self.prev_ps_weights
            updated = torch.ceil(diff.abs()).clamp(0, 1).bool()
            self.updated_since_init |= updated
            self.prev_ps_weights = g_ps_weights.clone()
            dload_participating = 4. * self.updated_since_init.sum()
            download_bytes = torch.zeros(self.num_clients)
            download_bytes[unique_clients] = dload_participating
        else:
            # we have to keep a full history of the ps weights
            # at every iteration, since a client could need to be
            # updated to the current ps weights from any arbitrary
            # previous iteration

            # the -1st entry of ps_weights_history is the current
            # ps weights. On the very first iteration, all clients
            # are entirely up-to-date, so they will have no download
            # cost. After that, client_num_stale_iters will never
            # have a zero in it
            self.ps_weights_history.append(g_ps_weights.clone().cpu())
            # figure out what the latest model weights each client
            # has. Clamping by maxlen/participation will
            # underestimate the number of bytes needed to download,
            # but as long as maxlen is large enough, this error
            # will be negligible
            maxlen = self.ps_weights_history.maxlen
            stale = self.client_num_stale_iters[unique_clients].clamp(
                    0, maxlen - 1
                )
            client_prev_weights = [self.ps_weights_history[-(s + 1)]
                                   for s in stale]
            # the ceil(...).sum() call just counts how many entries
            # g_ps_weights and prev disagree on -- this is the
            # number of new weights that the client needs to download
            # (it's the same as
            #  torch.where(g_ps_weights != prev)[0].numel(),
            #  but ~6x faster)
            ps_weights_cpu = g_ps_weights.cpu()
            download_bytes_participating = 4 * torch.tensor(
                    [torch.ceil(
                        (ps_weights_cpu - prev).abs()
                     ).clamp(0, 1).sum()
                     for prev in client_prev_weights]
                )
            download_bytes = torch.zeros(self.num_clients)
            download_bytes[unique_clients] = download_bytes_participating
            self.client_num_stale_iters[unique_clients] = 0
            self.client_num_stale_iters += 1

        upload_bytes_participating = {
                    "uncompressed": self.args.grad_size,
                    "true_topk": self.args.grad_size,
                    "local_topk": self.args.k,
                    "sketch": self.args.num_rows * self.args.num_cols,
                    "fedavg": self.args.grad_size
                }[self.args.mode] * 4
        upload_bytes = torch.zeros(self.num_clients)
        upload_bytes[unique_clients] = upload_bytes_participating

        #for q, batches in zip(self.batches_queues, proc_batches):
        #    q.put(batches)
        # now every process has the batches assigned to it
        queue_idxs = []
        for i, batches in enumerate(proc_batches):
            queue_idx = i % len(self.batches_queues)
            self.batches_queues[queue_idx].put(batches)
            #print("inserted into ", queue_idx)
            queue_idxs.append(queue_idx)
        #print("queue_idxs", queue_idxs)

        # get results from each process (which have already been aggregated
        # over the batches we gave to that process)
        results = []
        for queue_idx in set(queue_idxs):
            #print("dequeue from ", queue_idx)
            q = self.results_queues[queue_idx]
            results.extend(q.get())

        ## and collect results
        #results = []
        #for results_queue in self.results_queues:
        #    r = results_queue.get()
        #    results.extend(r)

        if self.args.mode == "sketch":
            shape = (self.args.num_rows, self.args.num_cols)
        elif self.args.mode in ["uncompressed", "true_topk", "local_topk",
                                "fedavg"]:
            shape = (self.args.grad_size,)

        # reduce the gradients
        transmit = torch.zeros(shape).to(self.args.device).float()
        #print("before reduce", shms())
        torch.distributed.reduce(transmit, 0)
        #print("after reduce", shms())

        g_minibatch_gradient[:] = transmit / batch[0].size()[0]

        split = split_results(results, self.args.num_results_train)
        return split + [download_bytes, upload_bytes]

    def _call_val(self, batch):
        split = [t.split(self.args.valid_batch_size) for t in batch]
        num_shards = len(split[0])
        batch_shards = [tuple(l[i] for l in split)
                        for i in range(num_shards)]

        per_proc = len(batch_shards) // len(self.update_forward_grad_ps)
        if per_proc == 0:
            per_proc = 999
        proc_batches = [batch_shards[i:i + per_proc]
                        for i in range(0, len(batch_shards), per_proc)]

        queue_idxs = []
        for i, batches in enumerate(proc_batches):
            queue_idx = i % len(self.batches_queues)
            self.batches_queues[queue_idx].put(batches)
            #print("inserted into ", queue_idx)
            queue_idxs.append(queue_idx)
        print("queue_idxs", queue_idxs)

        # get results from each process (which have already been aggregated
        # over the batches we gave to that process)
        results = []
        for queue_idx in set(queue_idxs):
            print("dequeue from ", queue_idx)
            q = self.results_queues[queue_idx]
            results.extend(q.get(timeout=10))
        return split_results(results, self.args.num_results_val)

    def __call__(self, batch):
        if self.training:
            return self._call_train(batch)
        else:
            return self._call_val(batch)

    def __getattr__(self, name):
        if name in ["parameters", "state_dict"]:
            global g_ps_weights
            set_param_vec(self.model, g_ps_weights.cpu())
            return getattr(self.model, name)

    def zero_grad(self):
        warnings.warn("workers already zero out their gradient by " +
                      "necessity before every forward pass")
        self.model.zero_grad()

class FedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, args):
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
        elif args.mode in ["true_topk", "local_topk", "fedavg",
                           "uncompressed"]:
            shape = (args.grad_size,)

        device = args.device
        self.Vvelocity = torch.zeros(shape).to(device)
        self.Verror = torch.zeros(shape).to(device)

    def get_lr(self):
        # return a scalar if all params have the same LR
        if len(self.param_groups) == 1:
            return self.param_groups[0]["lr"]

        # if there are multiple param groups, then each group may
        # have a different learning rate
        lr_vec = []
        for group in self.param_groups:
            lr = group["lr"]
            group_len = 0
            for p in group["params"]:
                if p.requires_grad:
                    group_len += p.numel()
            ones = torch.ones(group_len, device=self.args.device).float()
            lr_vec.append(ones * lr)
        return torch.cat(lr_vec).to(self.args.device)

    def step(self):
        global g_ps_weights
        global g_minibatch_gradient
        global g_lr

        lr = self.get_lr()

        if ((isinstance(lr, float) and lr == 0) or
            (isinstance(lr, torch.Tensor) and lr.abs().sum() == 0)):
            print("WARNING: LR is 0")

        # update g_lr so the model can use it next time for fedavg
        if self.args.mode == "fedavg":
            # only support scalar lr for fedavg
            assert isinstance(lr, float)
            g_lr[:] = lr

        weight_update, new_Vvelocity, new_Verror = get_server_update(
                g_minibatch_gradient,
                self.Vvelocity,
                self.Verror,
                self.args,
                1 if self.args.mode == "fedavg" else lr)

        # update ps_weights, momentums, and errors
        g_ps_weights -= weight_update

        self.Vvelocity[:] = new_Vvelocity
        self.Verror[:] = new_Verror

    def zero_grad(self):
        raise NotImplementedError("Please call zero_grad() on the model instead")


def args2sketch(args):
    return CSVec(d=args.grad_size, c=args.num_cols,
                 r=args.num_rows, device=args.device,
                 numBlocks=args.num_blocks)

def get_server_update(gradient, Vvelocity, Verror, args, lr):
    helper = {"sketch": _server_helper_sketched,
              "local_topk": _server_helper_local_topk,
              "true_topk": _server_helper_true_topk,
              "fedavg": _server_helper_fedavg,
              "uncompressed": _server_helper_uncompressed,
             }[args.mode]

    weight_update, new_Vvelocity, new_Verror = helper(
            gradient, Vvelocity, Verror, args, lr
        )

    return weight_update, new_Vvelocity, new_Verror

def _server_helper_fedavg(avg_update, Vvelocity, Verror, args, lr):
    assert args.error_type == "none"
    assert args.local_momentum == 0
    assert lr == 1

    rho = args.virtual_momentum
    torch.add(avg_update,
              Vvelocity,
              alpha=rho,
              out=Vvelocity)
    ps_update = Vvelocity

    return ps_update, Vvelocity, Verror

def _server_helper_uncompressed(gradient, Vvelocity, Verror, args, lr):

    rho = args.virtual_momentum
    torch.add(gradient,
              Vvelocity,
              alpha=rho,
              out=Vvelocity)
    grad = Vvelocity
    if args.do_dp and args.dp_mode == "server":
        #grad = clip_grad(args.l2_norm_clip, grad)
        noise = torch.normal(mean=0, std=args.noise_multiplier, size=grad.size()).to(args.device)
        grad += noise
    return grad * lr, Vvelocity, Verror

def _server_helper_true_topk(gradient, Vvelocity, Verror, args, lr):
    assert args.error_type == "virtual"

    rho = args.virtual_momentum

    # Vvelocity = rho * Vvelocity + gradient
    torch.add(gradient,
              Vvelocity,
              alpha=rho,
              out=Vvelocity)
    Verror += Vvelocity

    update = _topk(Verror, k=args.k)

    # we need to do momentum factor masking on the worker
    # momentum vectors for true_topk, which we can't do in
    # the worker because we don't know the global topk yet
    global g_participating_clients
    global g_client_velocities
    if args.local_momentum > 0:
        rows = g_participating_clients.view(-1,1)
        nz = update.nonzero()[:,0]
        g_client_velocities[rows, nz].zero_()


    # error feedback
    Verror[update.nonzero()] = 0

    # momentum factor masking
    Vvelocity[update.nonzero()] = 0

    return update * lr, Vvelocity, Verror

def _server_helper_local_topk(local_topk_grad, Vvelocity, Verror, args, lr):
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
    torch.add(local_topk_grad, Vvelocity, alpha=rho, out=Vvelocity)
    # ignore Verror, since you can't do virtual error with local topk

    # and no need to do momentum factor masking for the virtual
    # momentum, since that would zero out the entire Vvelocity every iter

    update = Vvelocity

    return update * lr, Vvelocity, Verror

def _server_helper_sketched(sketched_grad, Vvelocity, Verror, args, lr):
    rho = args.virtual_momentum
    k = args.k

    # must do the same type of momentum as error accumulation
    if args.error_type == "local":
        assert args.virtual_momentum == 0
    elif args.error_type == "virtual":
        assert args.local_momentum == 0

    torch.add(sketched_grad, Vvelocity, alpha=rho, out=Vvelocity)
    if args.error_type == "local":
        Verror = Vvelocity
    elif args.error_type == "virtual":
        Verror += Vvelocity

    sketch = args2sketch(args)
    sketch.accumulateTable(Verror)
    #noise = torch.normal(mean=0, std=args.noise_multiplier, size=[args.grad_size]).to(args.device)
    #sketch.accumulateVec(noise)
    #noise = torch.normal(mean=0, std=args.noise_multiplier, size=[args.rows, args.cols]).to(args.device)
    #sketch.accumulateTable(noise)
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

def split_results(results, n_results):
    return [np.array([r[i] for r in results]) for i in range(n_results)]
