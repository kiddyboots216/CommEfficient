import torch
import numpy as np
import ctypes
from utils import get_param_vec, set_param_vec, get_grad, _topk
import copy
import os
import time
import torch.multiprocessing as multiprocessing
from csvec import CSVec
import torch.distributed as dist
import queue

def update_forward_grad_loop(input_model, ps_weights, client_weights,
                             client_errors, client_velocities,
                             batches_queue, results_queue,
                             rank, world_size,
                             compute_loss_train, compute_loss_val, args):
    torch.cuda.set_device(rank - 1)

    model = input_model.to(args.device)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "5315"
    torch.distributed.init_process_group("nccl", rank=rank,
                                         world_size=world_size)
    while True:
        try:
            batches = batches_queue.get(timeout=30)
        except queue.Empty:
            print("batch queue was empty")
            return
        if batches is None:
            # reached the end of training
            break

        # get the latest weights from the parameter server
        local_ps_weights = ps_weights.to(args.device)

        # sum the gradient over all batches
        if args.mode in ["uncompressed", "true_topk", "local_topk"]:
            shape = (args.grad_size,)
        elif args.mode == "sketch":
            shape = (args.num_rows, args.num_cols)
        sum_g = torch.zeros(shape).to(args.device).float()

        # first batch, first tensor (client_indices), first datum
        is_train = batches[0][0][0] != -1

        all_results = []
        for batch in batches:
            g, results = process_batch(
                    batch, model, local_ps_weights, client_weights,
                    client_errors, client_velocities,
                    compute_loss_train, compute_loss_val, args
                )

            if is_train:
                sum_g += g
            all_results.append(results)

        results_queue.put(all_results)

        if is_train:
            # reduce the locally summed g across devices
            torch.distributed.barrier()
            torch.distributed.reduce(sum_g, 0)

def process_batch(batch, model, ps_weights, client_weights,
                  client_errors, client_velocities,
                  compute_loss_train, compute_loss_val, args):
        client_indices = batch[0]
        is_train = client_indices[0] != -1
        batch = batch[1:]
        batch = [t.to(args.device) for t in batch]
        assert (client_indices - client_indices[0]).abs().sum() == 0
        client_id = client_indices[0]

        # figure out what model weights this worker should use
        new_worker_weights = None
        if args.do_topk_down:
            worker_weights = client_weights[client_id].to(args.device)
            new_worker_weights = get_new_worker_weights(ps_weights,
                                                        worker_weights,
                                                        args)
            new_worker_weights = new_worker_weights.to(args.device)
        else:
            new_worker_weights = ps_weights

        # get model ready
        set_param_vec(model, new_worker_weights)

        transmit = None
        if is_train:
            model.train()
            model.zero_grad()
            # get our client's local velocity & local error vectors
            velocity = None
            error = None
            if client_velocities is not None:
                velocity = client_velocities[client_id].to(args.device)
            if client_errors is not None:
                error = client_errors[client_id].to(args.device)

            results, transmit = local_step(model, velocity, error, batch,
                                           compute_loss_train, args)
        else:
            model.eval()
            results = forward_grad(model, batch, compute_loss_val, args,
                                   compute_grad=False)
        return transmit, results

def local_step(model, velocity, error, batch, compute_loss, args):
    # g is a (possibly compressed) gradient
    g, results = forward_grad(model, batch, compute_loss, args)

    # reduce the importance of this gradient if the batch size was smaller
    # than usual
    g = g * batch[0].size(0) / args.local_batch_size

    # if needed, do local momentum
    if args.local_momentum > 0:
        # this does velocity[:] = m * velocity + g, but twice as fast
        torch.add(g, velocity, alpha=args.local_momentum, out=velocity)

    # if needed, do local error correction
    if args.error_type == "local":
        error += velocity if velocity is not None else g
        to_transmit = error
    else:
        to_transmit = velocity if velocity is not None else g

    if args.mode == "local_topk":
        assert args.error_type == "local"
        # topk is impossibly slow on CPU, very fast on GPU
        to_transmit = _topk(to_transmit.to(args.device), k=args.k)
        nz = to_transmit.nonzero()
        # error feedback
        error[nz] = 0

        # if we're doing local momentum, do momentum factor masking
        if args.local_momentum > 0:
            velocity[nz] = 0

    return results, to_transmit

def get_new_worker_weights(ps_weights, worker_weights, args):
    device = args.device

    ps_weights = ps_weights.to(device)
    worker_weights = worker_weights.to(device)

    # we'll update the old worker_weights with a possibly compressed
    # version of diff_vec
    diff_vec = ps_weights - worker_weights
    if args.do_topk_down:
        weight_update = _topk(diff_vec, k=args.k)
    else:
        weight_update = diff_vec

    new_worker_weights = worker_weights + weight_update
    #print(f"{torch.norm(weight_update, 2)}")
    #print(f"{updated_vec} = {client_weights} + {weight_update}")
    return new_worker_weights

def forward_grad(model, batch, compute_loss, args, compute_grad=True):
    device = args.device

    # divide up batch (for gradient accumulation when memory constrained)
    num_shards = args.num_train_batch_shards
    microbatch_size = args.local_batch_size // num_shards
    batch_size = batch[0].size()[0]

    # accumulators for the loss & metric values
    accum_loss = 0
    accum_metrics = None
    # need the max(1, ...) since the last batch in an epoch might be small

    num_iters = max(1, batch_size // microbatch_size)
    for i in range(num_iters):
        # extract current microbatch
        start = i * microbatch_size
        end = (i+1) * microbatch_size
        microbatch = [t[start:end] for t in batch]

        # forward pass
        loss, *metrics = compute_loss(model, microbatch, args)

        # if first time through, we find out how many metrics there are
        if accum_metrics is None:
            accum_metrics = [0 for _ in metrics]

        # accumulate loss & metrics
        accum_loss += loss.item()
        for i, m in enumerate(metrics):
            accum_metrics[i] += m.item()

        # backward pass
        if compute_grad:
            loss.backward()

    # gradient clipping
    if compute_grad and args.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       args.max_grad_norm)

    # "average" here is over the gradient accumulation steps
    average_loss = accum_loss / num_shards
    average_metrics = [m / num_shards for m in accum_metrics]

    results = [average_loss] + average_metrics

    if not compute_grad:
        return results

    grad = get_grad(model, args)

    # compress the gradient if needed
    if args.mode == "sketch":
        sketch = CSVec(d=args.grad_size, c=args.num_cols,
            r=args.num_rows, device=device,
            numBlocks=args.num_blocks)
        sketch.accumulateVec(grad)
        g = sketch.table
    elif args.mode == "true_topk":
        g = grad
    elif args.mode == "local_topk":
        # ideally we'd return the compressed version of the gradient,
        # i.e. _topk(grad, k=args.k). However, for sketching we do momentum
        # in the sketch, whereas for topk we do momentum before taking topk
        # so we have to return an inconsistent quantity here
        g = grad
    elif args.mode == "localSGD":
        # TODO: scheduling LR doesn't work
        grad *= args.lr_scale
        weights -= grad
        args.num_local_iters -= 1
        if args.num_local_iters > 0:
            g_recursive, results_recursive = forward_grad(model, weights, batch,
                    criterion, metric, args)
            g = grad + g_recursive
            results = [r + r_recursive for (r, r_recursive) in zip(results, results_recursive)]
        else:
            g = grad
    elif args.mode == "uncompressed":
        g = grad

    return g, results
