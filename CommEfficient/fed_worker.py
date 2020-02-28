import torch
import numpy as np
import ctypes
from utils import get_param_vec, set_param_vec, get_grad, _topk, clip_grad
import copy
import os
import time
import math
import torch.multiprocessing as multiprocessing
from csvec import CSVec
import torch.distributed as dist
import queue

def worker_loop(input_model, ps_weights, 
                #client_weights, 
                weight_update,
                client_errors,
                client_velocities, batches_queue, results_queue, fedavg_lr,
                rank, world_size, compute_loss_train, compute_loss_val,
                compute_loss_mal, args):
    torch.cuda.set_device(rank - 1)

    model = input_model.to(args.device)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.port)
    torch.distributed.init_process_group("nccl", rank=rank,
                                         world_size=world_size)
    while True:
        try:
            # batches is a list of batches that we should process
            # as if we were different workers for each batch
            # each batch in batches will have data belonging to a
            # single client (asserted in process_batch)
            batches = batches_queue.get(timeout=120)
        except queue.Empty:
            print("batch queue was empty")
            return
        if batches is None:
            print("done training")
            # reached the end of training
            break

        # get the latest weights from the parameter server
        local_ps_weights = ps_weights.clone().to(args.device)
        local_weight_update = None
        if args.do_mal_forecast:
            # get the last weight update if necessary
            local_weight_update = weight_update.clone().to(args.device)

        # sum the gradient over all batches
        if args.mode in ["uncompressed", "true_topk",
                         "local_topk", "fedavg"]:
            shape = (args.grad_size,)
        elif args.mode == "sketch":
            shape = (args.num_rows, args.num_cols)
        sum_g = torch.zeros(shape).to(args.device).float()
        sum_g_maybe_mal = torch.zeros(shape).to(args.device).float()

        # first batch, first tensor (client_indices), first datum
        is_train = batches[0][0][0] != -1

        # this is the starting learning rate (which possibly decays) when
        # carrying out fedavg
        lr = fedavg_lr.to(args.device)

        all_results = []
        # loop over workers we have to process (see comment above)
        for batch in batches:
            if args.mode == "fedavg" and is_train:
                assert args.error_type == "none"
                assert args.local_momentum == 0

                original_ps_weights = local_ps_weights.clone()
                # split "batch", which is this client's entire dataset,
                # into smaller batches to run local SGD on
                if args.fedavg_batch_size == -1:
                    local_batches = [batch]
                    n_batches = 1
                else:
                    local_batches = [torch.split(t, args.fedavg_batch_size)
                                     for t in batch]
                    n_batches = len(local_batches[0])
                    local_batches = [tuple(split[i]
                                           for split in local_batches)
                                     for i in range(n_batches)]

                n_steps = n_batches * args.num_fedavg_epochs
                step = 0
                accum_results = None
                for epoch in range(args.num_fedavg_epochs):
                    for batch in local_batches:
                        g, results = process_batch(
                                batch, model, local_ps_weights,
                                local_weight_update,
                                client_errors, client_velocities,
                                compute_loss_train, compute_loss_val, 
                                compute_loss_mal, args
                            )
                        if accum_results is None:
                            accum_results = results
                        else:
                            # accumulate results
                            for i in range(len(accum_results)):
                                accum_results[i] += results[i]
                        # g is the sum of gradients over examples, but
                        # we need to update the model with the avg grad
                        g /= batch[0].size()[0]
                        decay = args.fedavg_lr_decay ** step
                        local_ps_weights -= g * lr * decay
                        step += 1
                # compute average results from accum_results
                results = [r / n_steps for r in accum_results]
                g = original_ps_weights - local_ps_weights
                # weight by the batch size (which in the case of fedavg
                # is the client's dataset size) so that clients without
                # much data are downweighted
                g *= batch[0].size()[0]

                # reset local_ps_weights so that if this process has
                # to process another worker batch, the next worker
                # starts from the correct weights
                local_ps_weights[:] = original_ps_weights[:]

            else:
                # for all non-fedavg modes, we just do a single step
                if args.do_test:
                    # daniel says don't commit debugging code but i don't want to type this out everytime 
                    if is_train:
                        g, results = torch.ones(args.grad_size).to(args.device), tuple(1.0 for _ in range(args.num_results_train))
                    else:
                        g, results = torch.ones(args.grad_size).to(args.device), tuple(1.0 for _ in range(args.num_results_val))
                else:
                    g, results = process_batch(
                            batch, model, local_ps_weights, local_weight_update,
                            client_errors, client_velocities,
                            compute_loss_train, compute_loss_val, 
                            compute_loss_mal, args
                        )

            if is_train:
                sum_g += g
                g_maybe_mal = torch.zeros_like(g)
                client_indices = batch[0]
                do_malicious = args.do_malicious and client_indices[0] in args.mal_ids
                if do_malicious:
                    g_maybe_mal = g
                sum_g_maybe_mal += g_maybe_mal
            all_results.append(results)

        results_queue.put(all_results)

        if is_train:
            # reduce the locally summed g across devices
            torch.distributed.reduce(sum_g, 0)
            torch.distributed.reduce(sum_g_maybe_mal, 0)

def process_batch(batch, model, ps_weights, weight_update,
                  client_errors, client_velocities,
                  compute_loss_train, compute_loss_val, compute_loss_mal, args):
        client_indices = batch[0]
        is_train = client_indices[0] != -1
        if is_train:
            cur_epoch = batch[-1][0]
            batch = batch[1:-1]
        else:
            batch = batch[1:]
        batch = [t.to(args.device) for t in batch]
        assert (client_indices - client_indices[0]).abs().sum() == 0
        client_id = client_indices[0].numpy()
        do_malicious = args.do_malicious and client_id in args.mal_ids and cur_epoch >= args.mal_epoch
        #print(f"comparing {client_id} and {args.mal_ids}: {client_id in args.mal_ids}")
        if do_malicious:
            print("being malicious", torch.bincount(batch[-1]))
            compute_loss_train = compute_loss_mal
            #args.mal_boost = cur_epoch * 1.0

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

        if args.do_mal_forecast:
            # this is our "dynamics model"
            new_worker_weights -= weight_update

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

    # locally, we need to deal with the sum of gradients across
    # examples, since we will torch.distributed.reduce the to_transmits,
    g *= batch[0].size(0)

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
        assert args.error_type in ["local", "none"]
        # topk is impossibly slow on CPU, very fast on GPU
        to_transmit = _topk(to_transmit.to(args.device), k=args.k)

        nz = to_transmit.nonzero()
        if error is not None:
            # error feedback
            error[nz] = 0

        # if we're doing local momentum, do momentum factor masking
        if args.local_momentum > 0:
            velocity[nz] = 0

    # sketched sgd with local error accumulation doesn't really make
    # sense, since when we send a sketch we don't know what portion
    # of the sketch is the "error"
    if error is not None:
        assert args.mode not in ["sketch", "uncompressed"]

    # we want to do momentum factor masking for all the compression
    # methods, but that's not possible to do for sketching, since
    # it's unknown which coordinates to mask out
    if velocity is not None:
        assert args.mode != "sketch"

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
    return new_worker_weights

def forward_grad(model, batch, compute_loss, args, compute_grad=True):
    device = args.device

    # divide up batch (for gradient accumulation when memory constrained)
    #num_shards = args.num_train_batch_shards
    # need the max(1, ...) since the last batch in an epoch might be small
    #microbatch_size = max(1, batch[0].size()[0] // num_shards)
    if args.microbatch_size > 0:
        microbatch_size = min(batch[0].size()[0], args.microbatch_size)
    else:
        microbatch_size = batch[0].size()[0]

    # accumulators for the loss & metric values
    accum_loss = 0
    accum_metrics = None

    num_iters = math.ceil(batch[0].size()[0] / microbatch_size)
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

        # accumulate loss & metrics, weighted by how many data points
        # were actually used
        accum_loss += loss.item() * microbatch[0].size()[0]
        for i, m in enumerate(metrics):
            accum_metrics[i] += m.item() * microbatch[0].size()[0]

        # backward pass
        if compute_grad:
            loss.backward()

    # gradient clipping
    if compute_grad and args.max_grad_norm is not None and args.mode not in ["sketch"]:
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       args.max_grad_norm * num_iters)

    # "average" here is over the data in the batch
    average_loss = accum_loss / batch[0].size()[0]
    average_metrics = [m / batch[0].size()[0] for m in accum_metrics]

    results = [average_loss] + average_metrics

    if not compute_grad:
        return results

    grad = get_grad(model, args)
    if args.do_dp:
        grad = clip_grad(args.l2_norm_clip, grad)
        if args.dp_mode == "worker":
            noise = torch.normal(mean=0, std=args.noise_multiplier, size=grad.size()).to(args.device)
            noise *= np.sqrt(args.num_workers)
            grad += noise

    # compress the gradient if needed
    if args.mode == "sketch":
        sketch = CSVec(d=args.grad_size, c=args.num_cols,
            r=args.num_rows, device=args.device,
            numBlocks=args.num_blocks)
        sketch.accumulateVec(grad)
        # gradient clipping
        if compute_grad and args.max_grad_norm is not None:
            sketch = clip_grad(args.max_grad_norm, sketch)
        g = sketch.table
    elif args.mode == "true_topk":
        g = grad
    elif args.mode == "local_topk":
        # ideally we'd return the compressed version of the gradient,
        # i.e. _topk(grad, k=args.k). However, for sketching we do momentum
        # in the sketch, whereas for topk we do momentum before taking topk
        # so we have to return an inconsistent quantity here
        g = grad
    elif args.mode == "fedavg":
        # logic for doing fedavg happens in process_batch
        g = grad
    elif args.mode == "uncompressed":
        g = grad

    return g, results

