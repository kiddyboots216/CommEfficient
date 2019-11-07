import torch
import numpy as np
import ctypes
from utils import get_param_vec, set_param_vec, get_grad, _topk
import copy
import multiprocessing
from csvec import CSVec

def init_pool(input_model, device, num_worker_gpus,
              worker_errors, worker_velocities,
              worker_transmitted, client_weights, ps_weights):
    global model
    global gw_ps_weights
    global gw_client_weights
    global gw_worker_errors
    global gw_worker_velocities
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
    gw_worker_velocities = worker_velocities
    gw_worker_errors = worker_errors
    gw_worker_transmitted = worker_transmitted

def zero_grad():
    global model
    model.zero_grad()

def update_forward_grad(worker_id, client_id,
                        batch, args, criterion, metric):

    # pull PS and client weights out of the shared memory block
    grad_size = args.grad_size
    num_clients = args.num_clients
    participation = args.participation

    global model
    global gw_ps_weights
    global gw_client_weights
    global gw_worker_velocities
    global gw_worker_errors
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

    # g is a (possibly compressed) gradient
    if args.model == "gpt2":
        f = forward_grad_gpt2
    else:
        f = forward_grad
    g, results = f(
            model, new_worker_weights,
            batch, criterion, metric, args
        )

    # figure out what to send, and store it in the transmitted
    # shared tensor in spot worker_id
    """
    elif args.mode == "local_topk":
        worker_topk_shape = (args.num_workers, args.k)
        worker_topk_i = sm2np(gw_worker_topk_i_sm, worker_topk_shape,
                              dtype=ctypes.c_long)
        worker_topk_v = sm2np(gw_worker_topk_v_sm, worker_topk_shape)
        worker_topk_i[worker_id,:] = g[1].cpu().numpy()[:]
        worker_topk_v[worker_id,:] = g[2].cpu().numpy()[:]

        # store the full gradient too (which is in g[0])
        # so the server can do error accumulation
        worker_grads_shape = (args.num_workers, args.grad_size)
        worker_grads = sm2np(gw_worker_grads_sm, worker_grads_shape)
        worker_grads[worker_id,:] = g[0].cpu().numpy()[:]
    """

    # get SM arrays as np arrays
    worker_velocity = gw_worker_velocities[client_id]
    worker_error = gw_worker_errors[client_id]
    transmitted = gw_worker_transmitted[worker_id]

    # do local momentum & error accumulation
    g = g.cpu()
    # this does worker_velocity[:] = m * worker_velocity + g, but twice
    # as fast
    torch.add(g, worker_velocity, alpha=args.local_momentum,
              out=worker_velocity)
    if args.error_type == "local":
        worker_error += worker_velocity
        to_transmit = worker_error
    else:
        to_transmit = worker_velocity

    if args.mode == "local_topk":
        # topk is impossibly slow on CPU, very fast on GPU
        to_transmit = _topk(to_transmit.to(args.device), k=args.k).cpu()
        nz = to_transmit.nonzero()
        # error feedback
        worker_error[nz] = 0
        # momentum factor masking
        worker_velocity[nz] = 0

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

def forward_grad(model, weights, batch,
                 criterion, metric, args):
    device = args.device
    model = model.to(device)
    model.train()
    weights = weights.to(device)
    set_param_vec(model, weights)
    batch = tuple(input_tensor.to(device) for input_tensor in batch)
    criterion = criterion.to(device)
    metric = metric.to(device)
    if args.is_supervised:
        ins, targets = batch
        outs = model(ins)
        results = compute_loss(outs, targets, criterion, metric, True, args)
    grad = get_grad(model, weights, args, train=True, device=device)

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

    return g, results

def forward_multiprocessed(batch, args, criterion, metric):
    grad_size = args.grad_size
    global model
    global gw_ps_weights
    device = args.device
    ps_weights = gw_ps_weights.to(args.device)
    model = model.to(device)
    model.eval()
    set_param_vec(model, ps_weights)
    batch = tuple(input_tensor.to(device) for input_tensor in batch)
    if args.is_supervised:
        criterion = criterion.to(device)
        metric = metric.to(device)
        ins, targets = batch
        outs = model(ins)
        results = compute_loss(outs, targets, criterion, metric, False, args)
    else:
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        microbatch_size = args.batch_size // (args.num_workers * args.num_train_batch_shards)
        accum_loss = None
        accum_acc = None
        val_batch_size = batch[3].size()[0]
        n_iters = val_batch_size // microbatch_size
        for i in range(n_iters):
            start = i * microbatch_size
            end = (i+1) * microbatch_size
            microbatch = [b[start:end] for b in batch]
            logits, labels = inference(model, microbatch, args)
            lm_logits, mc_logits = logits
            lm_labels, mc_labels = labels
            nll = criterion(lm_logits, lm_labels).detach().cpu().numpy()
            acc = accuracy(mc_logits, mc_labels)
            if accum_loss is not None:
                accum_loss += nll 
            else:
                accum_loss = nll 
            if accum_acc is not None:
                accum_acc += acc 
            else:
                accum_acc = acc
        results = accum_loss, accum_acc
    return results

def compute_loss(outs, targets, criterion, metric, train, args):
    num_clients = args.num_clients
    participation = args.participation
    n_workers = int(num_clients * participation)
    loss = criterion(outs, targets)
    if train:
        loss.backward()
    batch_loss = loss.cpu().detach().numpy()
    acc = metric(outs, targets).float().mean().cpu().detach().numpy()
    return batch_loss, acc

# GPT2 SPECIFIC FUNCTIONS

def forward_grad_gpt2(model, weights, batch, criterion,
        metric, args):
    device = args.device
    model.to(device)
    model.train()
    weights = weights.to(device)
    set_param_vec(model, weights)
    batch = tuple(input_tensor.to(device) for input_tensor in batch)
    microbatch_size = args.batch_size // (args.num_workers * args.num_train_batch_shards)
    train_batch_size = batch[3].size()[0]
    accum_loss = None
    n_iters = train_batch_size // microbatch_size
    for i in range(n_iters):
        start = i * microbatch_size
        end = (i+1) * microbatch_size
        microbatch = [b[start:end] for b in batch]
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = microbatch
        (lm_loss), (mc_loss), *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, lm_labels=lm_labels
        )
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.num_train_batch_shards
        #print(f"Loss: {loss} from {lm_loss} and {mc_loss}")
        loss.backward()
        # TODO: Make sure this is ok for GPT2
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm * args.num_workers)
        if accum_loss is not None:
            accum_loss += loss
        else:
            accum_loss = loss
        #print(f"accum loss: {accum_loss} from {loss}")
    #print(f"accum loss: {accum_loss}")
    grad = get_grad(model, weights, args, device=device)
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
        #topk_i = torch.topk(grad**2, args.k, sorted=False)[1]
        #topk_v = grad[topk_i]
        #g = _topk(grad, k=args.k)
        #g = (grad, topk_i, topk_v)
        g = grad
    if accum_loss is not None:
        loss = accum_loss.item()/max(args.num_train_batch_shards, 1)
    else:
        loss = 0
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


def inference(model, batch, args):
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
