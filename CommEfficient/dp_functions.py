class DPOptimizer(FedCommEffOptimizer):
    def __init__(self
                optimizer,
                args,
                dp_sum_query,
                num_microbatches=None):
        super().__init__(optimizer, args)
        self.dp_sum_query = dp_sum_query
        self.num_microbatches = num_microbatches
        self._summary_value = 0

        self._global_parameters = self.dp_sum_query.initial_global_state()
        self._derived_records_data = defaultdict(list)

    def step(self, indices):
        # just assume no momentum or error accumulation for now
        new_ps_weights = self.dp_get_updated_server(self.args, lr)

        # update ps_weights, momentums, and errors
        ps_weights = sm2np(g_ps_weights_sm, (self.args.grad_size,))
        ps_weights[:] = new_ps_weights


    def dp_get_updated_server(self, lr):
        return _dp_server_helper(args, lr)

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

        # Get the correct shape gradient tensors to then set to the initial
        # state of the sample. Often all zero for zero gradients.
        sample_state = self.dp_sum_query.initial_sample_state(
            nest.parameters_to_tensor_groups(param_groups, 'data')
        )

        # Get the parameters for doing the dp query on this sample of data
        sample_params = self.dp_sum_query.derive_sample_params(self._global_parameters)

        self._derived_records_data = defaultdict(list)

        grad = np.sum([torch.from_numpy(g).to(device) for g in worker_grads])
        grad /= args.num_workers
        record = torch.from_numpy(grad).to(device)
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
                    np.array(v) > self._global_parameters.l2_norm_clip.detach().numpy())
                self._summary_value = {"percentage_clipped": p_clip}

        # Finish the DP query, usually by adding noise to the accumulated gradient information
        final_grads, _ = self.dp_sum_query.get_noised_result(sample_state, self._global_parameters)
        return (ps_weights - final_grads * lr).cpu()

class DPGaussianOptimizer(DPOptimizer):
    """ Specific Gaussian mechanism optimizer for L2 clipping and noise privacy """

    def __init__(self,
                 l2_norm_clip,
                 noise_multiplier,
                 ledger=None,
                 *args,
                 **kwargs):
        dp_sum_query = GaussianDPQuery(l2_norm_clip, l2_norm_clip * noise_multiplier)

        if ledger:
            dp_sum_query = QueryWithLedger(dp_sum_query, ledger=ledger)

        super().__init__(
            dp_sum_query=dp_sum_query,
            *args,
            **kwargs
        )

    @property
    def ledger(self):
        return self.dp_sum_query.ledger

