import torch
import torch.nn as nn
import ray

from worker import Worker

class FedWorker(Worker):
	def __init__(self, dataloader, worker_index, model_maker, model_config, kwargs):
		super().__init__(worker_index, model_maker, model_config, kwargs)
		self.dataloader = dataloader

	def train(self, iterations=1):
		running_metrics = {metric: 0.0 for metric in self.metrics}
		for _ in range(iterations):
			metrics = self.train_epoch()
			for metric in metrics:
				running_metrics[metric] += metric
		running_metrics = {k: v/len(self.dataloader) for k,v in running_metrics.items()}
		return running_metrics

	def train_epoch(self):
		running_metrics = {metric: 0.0 for metric in self.metrics}
		for step, (x, y) in enumerate(self.dataloader):
			metrics = self.train_batch(step, x, y)
			for metric in metrics:
				running_metrics[metric] += metric
		running_metrics = {k: v/len(self.dataloader) for k,v in running_metrics.items()}
		return running_metrics

	def train_batch(self, x, y):
		# optimizer.zero_grad()
		self.zero_grad()
		# model.forward()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.sketchedModel(inputs)
        loss = self.criterion(outputs, targets)
        accuracy = self.correctCriterion(outputs, targets)
        # loss.backward()
        loss.sum().backward()
        # optimizer.step()
        self.step_number += 1
        self.param_groups[0].update(**self.param_values())
        gradVec = self._getGradVec()
        # weight decay
        if self.weight_decay != 0:
            gradVec.add_(self.weight_decay, self._getParamVec())
        if self.nesterov:
            self.u.add_(gradVec).mul_(self.momentum)
            self.v.add_(self.u).add_(gradVec)
        else:
            self.u.mul_(self.momentum).add_(gradVec)
            self.v.add_(self.u)
        self._setGradVec(gradVec * self._getLRVec())
        self._updateParamsWithGradVec()
        return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()