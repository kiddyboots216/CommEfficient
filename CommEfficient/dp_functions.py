from collections import defaultdict
from math import ceil

import numpy as np
import torch

from torchprivacy.dp_query import GaussianSumQuery
from torchprivacy.analysis import PrivacyLedger, QueryWithLedger
from torchprivacy.analysis import compute_noise_multiplier

class DPHook:
    def __init__(self,
            dp_sum_query,
            args):
        self.dp_sum_query = dp_sum_query
        self._summary_value = 0

        self._global_parameters = self.dp_sum_query.initial_global_state()
        self._derived_records_data = defaultdict(list)

    def client_hook(self, grad, args):
        # NOTE: Because we're doing client level DP, we don't need to
        # microbatch. This only works for client level DP.
        record = grad
        # Initial state of the sample is probably all zeros
        sample_state = self.dp_sum_query.initial_sample_state(record)

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
        return final_grads


class DPGaussianHook(DPHook):
    def __init__(self, args):
        if args.num_clients is None:
            args.num_clients = args.num_workers
        args.participation = args.num_workers / args.num_clients
        #args.noise_multiplier = compute_noise_multiplier(args)
        args.noise_multiplier = 1.0/80
        dp_sum_query = GaussianSumQuery(args.l2_norm_clip, args.noise_multiplier)

        if args.ledger:
            ledger = PrivacyLedger(args.num_clients, args.participation)
            dp_sum_query = QueryWithLedger(dp_sum_query, ledger=ledger)

        super().__init__(
            dp_sum_query=dp_sum_query,
            args=args,
        )

    @property
    def ledger(self):
        return self.dp_sum_query.ledger
