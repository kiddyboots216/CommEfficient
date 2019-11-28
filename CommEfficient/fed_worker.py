import torch
import numpy as np
import ctypes
from utils import get_param_vec, set_param_vec, get_grad, _topk
import copy
import torch.multiprocessing as multiprocessing
from csvec import CSVec

def init_pool(input_model, device, num_worker_gpus,
              client_errors, client_velocities,
              worker_transmitted, client_weights, ps_weights):
    global model
    global gw_ps_weights
    global gw_client_weights
    global gw_client_errors
    global gw_client_velocities
    global gw_worker_transmitted

    # use the first num_worker_gpus gpus
    if torch.cuda.is_available():
        process_id = multiprocessing.current_process()._identity[0]
        # just in case the process_ids aren't zero-indexed
        device_id = process_id % num_worker_gpus
        torch.cuda.set_device(device_id)

    model = copy.deepcopy(input_model)
    model.to(device)
    gw_ps_weights = ps_weights
    gw_client_weights = client_weights
    gw_client_velocities = client_velocities
    gw_client_errors = client_errors
    gw_worker_transmitted = worker_transmitted

def zero_grad():
    global model
    model.zero_grad()

def forward(batch, compute_loss, args):
    global model
    global gw_ps_weights

    model = model.to(args.device)
    model.eval()
    set_param_vec(model, gw_ps_weights.to(args.device))
    return forward_grad(model, batch, compute_loss, args,
                        compute_grad=False)

def update_forward_grad(worker_id, client_id, batch, compute_loss, args):

    zero_grad()
    # pull PS and client weights out of the shared memory block
    grad_size = args.grad_size
    num_clients = args.num_clients

    global model
    global gw_ps_weights
    global gw_client_weights
    global gw_client_velocities
    global gw_client_errors
    global gw_worker_transmitted

    # figure out what model weights this worker should use
    new_worker_weights = None
    if args.do_topk_down:
        worker_weights = gw_client_weights[client_id].to(args.device)
        new_worker_weights = get_new_worker_weights(ps_weights,
                                                    worker_weights,
                                                    args)
    else:
        ps_weights = gw_ps_weights.to(args.device)
        new_worker_weights = ps_weights

    # get model ready
    model = model.to(args.device)
    model.train()
    set_param_vec(model, new_worker_weights.to(args.device))

    # g is a (possibly compressed) gradient
    g, results = forward_grad(model, batch, compute_loss, args)

    # figure out what to send, and store it in the transmitted
    # shared tensor in spot worker_id
    # get our specific local velocity/local error/transmitted vectors
    velocity = None
    error = None
    if gw_client_velocities is not None:
        velocity = gw_client_velocities[client_id]
    if gw_client_errors is not None:
        error = gw_client_errors[client_id]
    transmitted = gw_worker_transmitted[worker_id]

    g = g.cpu()

    # if needed, do local momentum
    # this does velocity[:] = m * velocity + g,
    # but twice as fast
    if args.local_momentum > 0:
        torch.add(g, velocity, alpha=args.local_momentum,
                  out=velocity)

    # if needed, do local error correction
    if args.error_type == "local":
        error += velocity if velocity is not None else g
        to_transmit = error
    else:
        to_transmit = velocity if velocity is not None else g

    if args.mode == "local_topk":
        assert args.error_type == "local"
        # topk is impossibly slow on CPU, very fast on GPU
        to_transmit = _topk(to_transmit.to(args.device), k=args.k).cpu()
        nz = to_transmit.nonzero()
        # error feedback
        error[nz] = 0

        # if we're doing local momentum, do momentum factor masking
        if args.local_momentum > 0:
            velocity[nz] = 0

    transmitted[:] = to_transmit

    return results

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

def forward_grad(model, batch, compute_loss, args,
                 compute_grad=True):

    device = args.device

    # divide up batch (for gradient accumulation when memory constrained)
    batch = tuple(input_tensor.to(device) for input_tensor in batch)
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
            if args.max_grad_norm is not None:
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
        #g = _topk(grad, k=args.k)
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
