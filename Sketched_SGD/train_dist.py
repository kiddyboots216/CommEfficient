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
import math

from core import *
from parameter_server import ParameterServer
from worker import Worker

# ALL THE STUFF THAT BREAKS

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

class StatsLogger():
    def __init__(self, keys):
        self.stats = {k:[] for k in keys}

    def append(self, output):
        for k,v in self.stats.items():
            v.append(output[k])
#             v.append(output[k].detach())

#     def stats(self, key):
#         return cat(*self._stats[key])

    def mean(self, key):
        return np.mean(self.stats[key], dtype=np.float)
#         return np.mean(to_numpy(self.stats(key)), dtype=np.float)

def run_batches(ps, workers, batches, minibatch_size, training):
    stats = StatsLogger(('loss', 'correct'))
#     model.train(training)
    for batchId, batch in enumerate(batches):
        inputs = batch["input"]
        targets = batch["target"]    
        input_minibatches = []
        target_minibatches = []
        batch_size = len(inputs)
        num_workers = len(workers)
        for i, _ in enumerate(workers):
            start = i * batch_size // num_workers
            end = (i+1) * batch_size // num_workers
            input_minibatches.append(inputs[start:end])
            target_minibatches.append(targets[start:end])
        if training:
            # workers do backward passes and calculate sketches
            losses, accuracies, sketches = list(zip(*ray.get([worker.forward.remote(
                input_minibatches[worker_id],
                target_minibatches[worker_id],
                training)
                for worker_id, worker in enumerate(workers)])))
            #losses, accuracies, sketches = list(zip(*ray.get([worker.forward.remote(
             #                                   inputs[int(worker_id * minibatch_size) : int((worker_id + 1) * minibatch_size)], 
              #                                  targets[int(worker_id * minibatch_size) : int((worker_id + 1) * minibatch_size)],
               #                                 training)
                #                            for worker_id, worker in enumerate(workers)])))
            # server initiates second round of communication
            hhcoords = ray.get(ps.compute_hhcoords.remote((sketches)))
#             workers answer, also giving the unsketched params
            topkAndUnsketched = list(zip(*ray.get([worker.send_topkAndUnsketched.remote(hhcoords) for worker in workers])))
            # server compute weight update, put it into ray
            weightUpdate = ray.get(ps.compute_update.remote(topkAndUnsketched))
            # workers apply weight update (can be merged with 1st line)
            ray.wait([worker.apply_update.remote(weightUpdate) for worker in workers])
        else:
#             pass
            losses, accuracies= list(zip(*ray.get([worker.forward.remote(
                input_minibatches[worker_id],
                target_minibatches[worker_id],
                training)
                for worker_id, worker in enumerate(workers)])))
            #losses, accuracies = list(zip(*ray.get([worker.forward.remote(
             #                                   inputs[int(worker_id * minibatch_size) : int((worker_id + 1) * minibatch_size)], 
              #                                  targets[int(worker_id * minibatch_size) : int((worker_id + 1) * minibatch_size)],
               #                                 training)
                #                            for worker_id, worker in enumerate(workers)])))
#         loss = criterion(outputs, targets)
#         nCorrect = correctCriterion(outputs, targets)
        iterationStats = {"loss": np.mean((losses)), "correct": np.mean((accuracies))}
#         if training:
#             loss.sum().backward()
#             optimizer.step()
        stats.append(iterationStats)
    return stats

def train_epoch(ps, workers, train_batches, test_batches, minibatch_size,
                timer, test_time_in_total=True):
    train_stats = run_batches(ps, workers, train_batches, minibatch_size, True)
    train_time = timer()
    test_stats = run_batches(ps, workers, test_batches, minibatch_size, False)
    test_time = timer(test_time_in_total)
    stats ={'train_time': train_time,
            'train_loss': train_stats.mean('loss'),
            'train_acc': train_stats.mean('correct'),
            'test_time': test_time,
            'test_loss': test_stats.mean('loss'),
            'test_acc': test_stats.mean('correct'),
            'total_time': timer.total_time}
    return stats

def train(ps, workers, train_batches, test_batches, epochs, minibatch_size,
          loggers=(), test_time_in_total=True, timer=None):
    timer = timer or Timer()
    for epoch in range(epochs):
        epoch_stats = train_epoch(ps, workers, train_batches, test_batches, minibatch_size,
                                  timer,
                                  test_time_in_total=test_time_in_total)
        lr = ray.get(workers[0].param_values.remote())['lr'] * train_batches.batch_size
        summary = union({'epoch': epoch+1, 'lr': lr}, epoch_stats)
#         track.metric(iteration=epoch, **summary)
        for logger in loggers:
            logger.append(summary)
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketched", action="store_true")
    parser.add_argument("--sketch_biases", action="store_true")
    parser.add_argument("--sketch_params_larger_than", action="store_true")
    parser.add_argument("-k", type=int, default=50000)
    parser.add_argument("--p2", type=int, default=4)
    parser.add_argument("--p1", type=int, default=0)
    parser.add_argument("--cols", type=int, default=500000)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=7)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=24)
    args = parser.parse_args()
    #args.batch_size = math.ceil(args.batch_size/args.num_workers) * args.num_workers
    model_maker = lambda model_config: Net(
         #{'prep': 4, 'layer1': 8,
          #                           'layer2': 16, 'layer3': 32}
    ).to(model_config["device"])
    model_config = {
    #     "device": "cpu",
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    }

    print('Downloading datasets')
    DATA_DIR = "sample_data"
    dataset = cifar10(DATA_DIR)

    lr_schedule = PiecewiseLinear([0, 5, args.epochs], [0, 0.4, 0])
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
    lr = lambda step: lr_schedule(step/len(train_batches))/args.batch_size
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
                            args.batch_size, shuffle=True,
                            set_random_choices=True, drop_last=True)
    test_batches = Batches(test_set, args.batch_size, shuffle=False,
                           drop_last=False)

    optim_args = {
        "k": args.k,
        "p2": args.p2,
        "p1": args.p1,
        "numCols": args.cols,
        "numRows": args.rows,
        "numBlocks": args.num_blocks,
        "lr": lr,
        "momentum": 0.9,
        "weight_decay": 5e-4*args.batch_size,
        "nesterov": True,
        "dampening": 0,
    }

    # warmed_up = False
    # while not warmed_up:
    #     try:
    #         for size in [batch_size, len(dataset['test']['labels']) % batch_size]:
    #                 warmup_cudnn(model_maker(model_config), size)
    #         warmed_up = True
    #     except RuntimeError as e:
    #         print(e)
    ray.init(ignore_reinit_error=True)
    num_workers = args.num_workers
    minibatch_size = args.batch_size/num_workers
    print(f"Passing in args {optim_args}")
    ps = ParameterServer.remote(model_maker, model_config, optim_args)
    # ps = ParameterServer(model_maker, model_config, optim_args)
    # Create workers.
    workers = [Worker.remote(worker_index, model_maker, model_config, optim_args) for worker_index in range(num_workers)]
    # workers = [Worker(worker_index, model_maker, model_config, optim_args) for worker_index in range(num_workers)]

    # track_dir = "sample_data"
    # with track.trial(track_dir, None, param_map=vars(optim_args)):

    train(ps, workers, train_batches, test_batches, args.epochs, minibatch_size,
          loggers=(TableLogger(), TSV), timer=timer,
          test_time_in_total=False)
