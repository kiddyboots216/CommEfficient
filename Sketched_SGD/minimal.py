import ray
import time
import torch
@ray.remote(num_gpus=1)
class Counter(object):
    def __init__(self):
        self.counter = torch.zeros(10).cuda()
        print("hello world")
        time.sleep(5)
    def train(self):
        for i in range(10):
            time.sleep(5)
            self.counter += torch.ones(10).cuda()
            print(i)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sketched", action="store_true")
parser.add_argument("--sketch_biases", action="store_true")
parser.add_argument("--sketch_params_larger_than", action="store_true")
parser.add_argument("-k", type=int, default=50000)
parser.add_argument("--p2", type=int, default=1)
parser.add_argument("--p1", type=int, default=0)
parser.add_argument("--cols", type=int, default=500000)
parser.add_argument("--rows", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_blocks", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--epochs", type=int, default=24)
parser.add_argument("--test", action="store_true")
args = parser.parse_args()
optim_args = {
        "k": args.k,
        "p2": args.p2,
        "p1": args.p1,
        "numCols": args.cols,
        "numRows": args.rows,
        "numBlocks": args.num_blocks,
        "lr": lambda step: 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4*args.batch_size,
        "nesterov": args.nesterov,
        "dampening": 0,
        }
ray.init(num_gpus=8)
num_workers = 4
from worker import Worker
workers = [Worker.remote(num_workers, worker_index, optim_args) for worker_index in range(num_workers)]
ray.wait([worker.step.remote() for worker in workers])
#counters = [Counter.remote() for _ in range(4)]
#ray.wait([counter.train.remote() for counter in counters])
