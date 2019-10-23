import torch
import numpy as np
from utils import sm2np, get_param_vec, set_param_vec, get_grad, _topk
import copy
import multiprocessing
from csvec import CSVec

def init_pool(input_model, device, num_gpus,
              worker_Sgrads_sm, worker_grads_sm,
              client_weights_sm, ps_weights_sm):
    global model
    global gw_ps_weights_sm
    global gw_client_weights_sm
    global gw_worker_Sgrads_sm
    global gw_worker_grads_sm

    if torch.cuda.is_available():
        process_id = multiprocessing.current_process()._identity[0]
        # just in case the process_ids aren't zero-indexed
        device_id = process_id % num_gpus
        torch.cuda.set_device(device_id)

    model = copy.deepcopy(input_model)
    model.to(device)
    gw_ps_weights_sm = ps_weights_sm
    gw_client_weights_sm = client_weights_sm
    gw_worker_Sgrads_sm = worker_Sgrads_sm
    gw_worker_grads_sm = worker_grads_sm

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
    global gw_ps_weights_sm
    global gw_client_weights_sm
    global gw_worker_Sgrads_sm
    global gw_worker_grads_sm

    ps_weights = sm2np(gw_ps_weights_sm, (grad_size,))
    ps_weights = torch.from_numpy(ps_weights).to(args.device)
    if args.do_topk_down:
        client_weights = sm2np(gw_client_weights_sm, (num_clients, grad_size))
        worker_weights = client_weights[client_id]
        worker_weights = torch.from_numpy(worker_weights).to(args.device)
        new_worker_weights = get_new_worker_weights(ps_weights,
                                                    worker_weights,
                                                    args)
    else:
        new_worker_weights = ps_weights

    # g is a (possibly compressed) gradient
    device = args.device
    model = model.to(device)
    model.train()
    weights = new_worker_weights.to(device)
    set_param_vec(model, weights)
    batch = tuple(input_tensor.to(device) for input_tensor in batch)
    criterion = criterion.to(device)
    metric = metric.to(device)
    f = forward_grad
    """
    if args.model == "gpt2":
        f = forward_grad_gpt2
    """
    g, results = f(
            model, new_worker_weights,
            batch, criterion, metric, args
        )
    # write g to the shared memory grad array in spot worker_id
    if args.mode == "sketch":
        worker_Sgrads_shape = (args.num_workers, args.num_rows, args.num_cols)
        worker_Sgrads = sm2np(gw_worker_Sgrads_sm, worker_Sgrads_shape)
        worker_Sgrads[worker_id,:,:] = g.cpu().numpy()[:,:]
    elif args.mode in ["true_topk", "local_topk", "localSGD"]:
        worker_grads_shape = (args.num_workers, args.grad_size)
        worker_grads = sm2np(gw_worker_grads_sm, worker_grads_shape)
        worker_grads[worker_id,:] = g.cpu().numpy()[:]

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
    """
    device = args.device
    model = model.to(device)
    model.train()
    weights = weights.to(device)
    set_param_vec(model, weights)
    batch = tuple(input_tensor.to(device) for input_tensor in batch)
    criterion = criterion.to(device)
    metric = metric.to(device)
    """
    if args.is_supervised:
        ins, targets = batch
        outs = model(ins)
        results = compute_loss(outs, targets, criterion, metric, True, args)
    else:
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
        if accum_loss is not None:
            loss = accum_loss.item()/max(args.num_train_batch_shards, 1)
        else:
            loss = 0
        results = [loss]

    grad = get_grad(model, weights, args)

    # compress the gradient if needed
    if args.mode == "sketch":
        sketch = CSVec(d=args.grad_size, c=args.num_cols,
            r=args.num_rows, device=args.device,
            numBlocks=args.num_blocks)
        sketch.accumulateVec(grad)
        g = sketch.table.cpu()
        del sketch
    elif args.mode == "true_topk":
        g = grad
    elif args.mode == "local_topk":
        g = _topk(grad, k=args.k)
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
    global gw_ps_weights_sm
    ps_weights = sm2np(gw_ps_weights_sm, (grad_size,))
    device = args.device
    ps_weights = torch.from_numpy(ps_weights).to(args.device)
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
    """
    device = args.device
    model.to(device)
    model.train()
    weights = weights.to(device)
    set_param_vec(model, weights)
    batch = tuple(input_tensor.to(device) for input_tensor in batch)
    """
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
    grad = get_grad(model, weights, args)
    # compress the gradient if needed
    if args.mode == "sketch":
        sketch = CSVec(d=args.grad_size, c=args.num_cols,
            r=args.num_rows, device=args.device,
            numBlocks=args.num_blocks)
        sketch.accumulateVec(grad)
        g = sketch.table.cpu()
    elif args.mode == "true_topk":
        g = grad
    elif args.mode == "local_topk":
        g = _topk(grad, k=args.k)
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
