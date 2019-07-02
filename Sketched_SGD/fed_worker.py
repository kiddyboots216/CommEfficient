import copy

import torch
import torch.nn as nn
import torch.optim as optim
import ray

from minimal import Correct, CSVec, SketchedModel, Net

@ray.remote(num_gpus=1.0)
class FedWorker(object):
    def __init__(self, dataloader, worker_index, kwargs):
        self.hp = kwargs
        print(f"Initializing worker {self.worker_index}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # we need to load the data in
        self.dataloader = dataloader
        # make the model with config (hardcoded for now)
        # self.model = model_maker(model_config)
        # uniform initialization?
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        self.model = Net({'prep': 1, 'layer1': 1,
                                 'layer2': 1, 'layer3': 1}).to(self.device)
        torch.random.set_rng_state(rand_state)
        # sketch the model
        self.model = SketchedModel(self.model)
        # we want an optimizer, it'll just be faster
        optimizer_object = getattr(optim, self.hp['optimizer'])
        optimizer_parameters = {k : v for k, v in self.hp.items() if k in optimizer_object.__init__.__code__.co_varnames}
        self.opt = optimizer_object(self.model.parameters(), **optimizer_parameters)
        # we need to make our own criterion, this can be cool in future
        criterion_object = getattr(nn, self.hp['criterion'])
        criterion_parameters = {k : v for k, v in self.hp.items() if k in criterion_object.__init__.__code__.co_varnames}
        self.criterion = criterion_object(**criterion_parameters).to(self.device)
        # make the accuracy
        self.accuracy = Correct().to(self.device)
        # set up the metrics
        self.metrics = self.hp['metrics']
        # make a CSVec and set self.sketched_model
        self._sketcher_init(self.hp['sketched'])

    def model_diff(self, opt_copy):
        diff_vec = []
        for group_id, param_group in enumerate(self.opt.param_groups):
            for idx, p in enumerate(param_group['params']):
                # calculate the difference between the current model and the stored model
                diff_vec.append(p.data.view(-1).float() - opt_copy.param_groups[group_id]['params'][idx].data.view(-1).float())
                # reset the current model to the stored model for later
                p.data = opt_copy.param_groups[group_id]['params'][idx].data
        self.diff_vec = torch.cat(diff_vec).to(self.device)
        # print(diff_vec)
        masked_diff = self.diff_vec[self.sketch_mask]
        # sketch the gradient
        self.sketch.zero()
        self.sketch += masked_diff
        del masked_diff
        # communicate only the table
        return self.sketch.table

    def send_topkAndUnsketched(self, hhcoords):
        # directly send whatever wasn't sketched
        unsketched = self.diff_vec[~self.sketch_mask]
        # COMMUNICATE
        return self.diff_vec[hhcoords], unsketched

    def apply_update(self, weightUpdate):
        start = 0
        for param_group in self.opt.param_groups:
            for p in param_group['params']:
                end = start + torch.numel(p)
                p.data = weightUpdate[start:end].reshape(p.data.shape)
                start = end

    def forward(self, test_batches):
        running_metrics = {metric: 0.0 for metric in self.metrics}
        for idx, batch in enumerate(test_batches):
            loss, acc = self._forward_batch(batch)
            running_metrics['loss'] += loss
            running_metrics['acc'] += acc
        running_metrics = {k: v/len(test_batches) for k,v in running_metrics.items()}
        return running_metrics

    def _forward_batch(self, test_batch):
        out = self.model(x)
        # calculate losses
        loss = self.criterion(out, y)
        # calculate metric
        acc = self.accuracy(out, y)
        return loss, acc

    def train(self, iterations=1):
        # we need to store a copy of the parameters
        opt_copy = copy.deepcopy(optim)
        running_metrics = {metric: 0.0 for metric in self.metrics}
        for _ in range(iterations):
            metrics = self.train_epoch()
            for k, v in metrics.items():
                running_metrics[k] += v
        # scale metrics appropriately
        running_metrics = {k: v/iterations for k,v in running_metrics.items()}
        return running_metrics, self.model_diff(opt_copy)

    def train_epoch(self):
        running_metrics = {metric: 0.0 for metric in self.metrics}
        for step, (x, y) in enumerate(self.dataloader):
            loss, acc = self.train_batch(step, x.cuda(), y.cuda())
            running_metrics['loss'] += loss
            running_metrics['acc'] += acc
            # for k, v in metrics.items():
            #   running_metrics[k] += v
        # scale metrics appropriately
        running_metrics = {k: v/len(self.dataloader) for k,v in running_metrics.items()}
        return running_metrics

    def train_batch(self, x, y):
        self.step_number += 1
        self.param_groups[0].update(**self.param_values())
        # zero the optimizer
        self.opt.zero_grad()
        # forward pass
        loss, acc = self._forward_batch(x)
        # backwards pass of loss
        loss.backwards()
        # step the optimizer
        self.opt.step()
        return loss, acc
        # return {'loss': loss.sum(), 'acc': acc} 

    def _sketcher_init(self, 
                 k=0, p2=0, numCols=0, numRows=0, numBlocks=1):
        #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids())) 
        #print(ray.get_gpu_ids())
        # set the sketched params as instance vars
        self.p2 = p2
        self.k = k
        # initialize sketchMask, sketch, momentum buffer and accumulated grads
        # this is D
        grad_size = 0
        sketchMask = []
        for group in self.opt.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    size = torch.numel(p)
                    if p.do_sketching:
                        sketchMask.append(torch.ones(size))
                    else:
                        sketchMask.append(torch.zeros(size))
                    grad_size += size
        self.grad_size = grad_size
        print(f"Total dimension is {self.grad_size} using k {self.k} and p2 {self.p2}")
        self.sketch_mask = torch.cat(sketchMask).byte().to(self.device)
        #print(f"sketchMask.sum(): {self.sketchMask.sum()}")
#         print(f"Make CSVec of dim{numRows}, {numCols}")
        self.sketch = CSVec(d=self.sketchMask.sum().item(), c=numCols, r=numRows, 
                            device=self.device, nChunks=1, numBlocks=numBlocks)