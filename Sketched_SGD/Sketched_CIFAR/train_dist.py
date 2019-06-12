from inspect import signature
from collections import namedtuple
import time
import numpy as np
import pandas as pd
from functools import singledispatch
from collections import OrderedDict
import track
import ray
import torch
import torch.nn as nn

from core import *
from sketched_sgd.parameter_server import ParameterServer
from sketched_sgd.worker import Worker

criterion = nn.CrossEntropyLoss(reduction='sum')
correctCriterion = Correct()

def run_batches(ps, workers, batches, training):
    stats = StatsLogger(('loss', 'correct'))
#     model.train(training)
    for batchId, batch in enumerate(batches):
        inputs = batch["input"]
        targets = batch["target"]
        if training:
            # workers do backward passes and calculate sketches
            losses, accuracies, sketches = list(zip(*ray.get([worker.forward.remote(
                                                criterion,
                                                correctCriterion,
                                                inputs[worker_id * minibatch_size : (worker_id + 1) * minibatch_size], 
                                                targets[worker_id * minibatch_size : (worker_id + 1) * minibatch_size],
                                                training)
                                            for worker_id, worker in enumerate(workers)])))
            # server initiates second round of communication
            hhcoords = ray.get(ps.compute_hhcoords.remote(sketches))
            # workers answer, also giving the unsketched params
            topkAndUnsketched = list(zip(*ray.get([worker.send_topkAndUnsketched.remote(hhcoords) for worker in workers])))
            # server compute weight update, put it into ray
            weightUpdate = ray.get(ps.compute_update.remote(topkAndUnsketched))
            # workers apply weight update (can be merged with 1st line)
            ray.get([worker.apply_update.remote(weightUpdate) for worker in workers])
        else:
            losses, accuracies = list(zip(*ray.get([worker.forward.remote(
                                        criterion,
                                        correctCriterion,
                                        inputs[worker_id * minibatch_size : (worker_id + 1) * minibatch_size], 
                                        targets[worker_id * minibatch_size : (worker_id + 1) * minibatch_size],
                                        training)
                                    for worker_id, worker in enumerate(workers)])))
#         loss = criterion(outputs, targets)
#         nCorrect = correctCriterion(outputs, targets)
        iterationStats = {"loss": np.mean(losses), "correct": np.mean(accuracies)}
#         if training:
#             loss.sum().backward()
#             optimizer.step()
        stats.append(iterationStats)
    return stats

def train_epoch(ps, workers, train_batches, test_batches,
                timer, test_time_in_total=True):
    train_stats = run_batches(ps, workers, train_batches, True)
    train_time = timer()
    test_stats = run_batches(ps, workers, test_batches, False)
    test_time = timer(test_time_in_total)
    stats ={'train_time': train_time,
            'train_loss': train_stats.mean('loss'),
            'train_acc': train_stats.mean('correct'),
            'test_time': test_time,
            'test_loss': test_stats.mean('loss'),
            'test_acc': test_stats.mean('correct'),
            'total_time': timer.total_time}
    return stats

def train(ps, workers, train_batches, test_batches, epochs,
          loggers=(), test_time_in_total=True, timer=None):
    timer = timer or Timer()
    for epoch in range(epochs):
        epoch_stats = train_epoch(ps, workers, train_batches, test_batches,
                                  timer,
                                  test_time_in_total=test_time_in_total)
        lr = workers[0].param_values()['lr'] * train_batches.batch_size
        summary = union({'epoch': epoch+1, 'lr': lr}, epoch_stats)
#         track.metric(iteration=epoch, **summary)
        for logger in loggers:
            logger.append(summary)
    return summary

if __name__ == "__main__":
    from easydict import EasyDict as edict
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_maker = lambda model_config: Net().to(device)
    model_config = {}
    # sketched_args = edict({
        
    # })

    print('Downloading datasets')
    DATA_DIR = "sample_data"
    dataset = cifar10(DATA_DIR)

    epochs = 24
    lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
    batch_size = 512
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
    lr = lambda step: lr_schedule(step/len(train_batches))/batch_size
    print('Starting timer')
    timer = Timer()

    print('Preprocessing training data')
    train_set = list(zip(
            transpose(normalise(pad(dataset['train']['data'], 4))),
            dataset['train']['labels']))
    print('Finished in {:.2f} seconds'.format(timer()))
    print('Preprocessing test data')
    test_set = list(zip(transpose(normalise(dataset['test']['data'])),
                        dataset['test']['labels']))
    print('Finished in {:.2f} seconds'.format(timer()))

    TSV = TSVLogger()

    train_batches = Batches(Transform(train_set, train_transforms),
                            batch_size, shuffle=True,
                            set_random_choices=True, drop_last=True)
    test_batches = Batches(test_set, batch_size, shuffle=False,
                           drop_last=False)
    optim_args = edict({
        "k": 50000,
        "p2": 4,
        "p1": 0,
        "numCols": 500000,
        "numRows": 5,
        "numBlocks": 1,
        "lr": lr,
        "momentum": 0.9,
        "weight_decay": 5e-4*batch_size,
        "nesterov": True,
        "dampening": 0,
    })
    ray.init(ignore_reinit_error=True)
    num_workers = 1
    ps = ParameterServer.remote(model_maker, model_config, optim_args)
    # Create workers.
    workers = [Worker.remote(worker_index, model_maker, model_config, optim_args) for worker_index in range(num_workers)]

    # track_dir = "sample_data"
    # with track.trial(track_dir, None, param_map=vars(optim_args)):
    train(ps, workers, train_batches, test_batches, epochs,
          loggers=(TableLogger(), TSV), timer=timer,
          test_time_in_total=False)