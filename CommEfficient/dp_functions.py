from collections import defaultdict
from math import ceil

import numpy as np
import torch

from utils import sm2np, get_param_vec, set_param_vec, get_grad, _topk
import copy
import multiprocessing
from csvec import CSVec

from functions import FedCommEffOptimizer
import pytorch_privacy.utils.torch_nest_utils as nest
from pytorch_privacy.dp_query import GaussianDPQuery, QueryWithLedger, PrivacyLedger

g_worker_Sgrads_sm = None
g_worker_grads_sm = None
g_client_weights_sm = None
g_ps_weights_sm = None

g_criterion = None
g_accuracy = None


class DPOptimizer(FedCommEffOptimizer):
    def __init__(self,
                optimizer,
                dp_sum_query,
                args):
        super().__init__(optimizer, args)
        self.dp_sum_query = dp_sum_query
        self.num_microbatches = args.num_microbatches
        self._summary_value = 0

        self._global_parameters = self.dp_sum_query.initial_global_state()
        self._derived_records_data = defaultdict(list)

    def step(self, indices):
        lr = self.get_lr()
        # just assume no momentum or error accumulation for now
        new_ps_weights = self.dp_get_updated_server(self.args, lr)

        # update ps_weights, momentums, and errors
        ps_weights = sm2np(g_ps_weights_sm, (self.args.grad_size,))
        ps_weights[:] = new_ps_weights


    def dp_get_updated_server(self, args, lr):
        return self._dp_server_helper(args, lr)

    def _dp_server_helper(self, args, lr):
        # assume that the worker did whatever microbatching was necessary
        # not going to implement that for now bc it's annoying
        
        global g_ps_weights_sm
        global g_worker_grads_sm

        device = torch.device(args.device)

        ps_weights = sm2np(g_ps_weights_sm, (args.grad_size,),)
        ps_weights = torch.from_numpy(ps_weights).to(device)

        worker_grads_shape = (args.num_workers, args.grad_size)
        worker_grads = sm2np(g_worker_grads_sm, worker_grads_shape)

        grad = np.sum([torch.from_numpy(g).to(device) for g in worker_grads])
        grad /= args.num_workers
        record = grad
        # Get the correct shape gradient tensors to then set to the initial
        # state of the sample. Often all zero for zero gradients.
        sample_state = self.dp_sum_query.initial_sample_state(
            record
            #nest.parameters_to_tensor_groups(grad, 'data')
        )

        # Get the parameters for doing the dp query on this sample of data
        sample_params = self.dp_sum_query.derive_sample_params(self._global_parameters)

        self._derived_records_data = defaultdict(list)

        # Accumulate the gradients onto the current sample stack, applying what ever DP operations are required
        sample_state = self.dp_sum_query.accumulate_record(sample_params, sample_state, record)
        # Gather any information of interest from the query
        derived_record_data = self.dp_sum_query.get_record_derived_data()

        for k, v in derived_record_data.items():
            self._derived_records_data[k].append(v)

        self._derived_records_data = dict(self._derived_records_data)

        for k, v in self._derived_records_data.items():
            # summarise statistics instead
            self._derived_records_data[k] = np.percentile(np.array(v), [10.0, 30.0, 50.0, 70.0, 90.0])
            if k == "l2_norm:":
                p_clip = np.mean(
                    np.array(v) > self._global_parameters.l2_norm_clip.detach().cpu().numpy())
                self._summary_value = {"percentage_clipped": p_clip}

        # Finish the DP query, usually by adding noise to the accumulated gradient information
        final_grads, _ = self.dp_sum_query.get_noised_result(sample_state, self._global_parameters)
        return (ps_weights - final_grads * lr).cpu()

class DPGaussianOptimizer(DPOptimizer):
    """ Specific Gaussian mechanism optimizer for L2 clipping and noise privacy """

    def __init__(self,
                 *args,
                 **kwargs):
        args,optimizer = args
        dp_sum_query = GaussianDPQuery(args.l2_norm_clip, args.l2_norm_clip * args.noise_multiplier)

        if args.ledger:
            ledger = PrivacyLedger(args.num_data, args.batch_size/args.num_data)
            dp_sum_query = QueryWithLedger(dp_sum_query, ledger=ledger)

        super().__init__(
            dp_sum_query=dp_sum_query,
            optimizer=optimizer,
            args=args,
            #*args,
            #**kwargs
        )

    @property
    def ledger(self):
        return self.dp_sum_query.ledger

