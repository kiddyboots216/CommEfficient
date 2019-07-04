import torch
import torch.nn as nn
import torch.optim as optim
import copy

from minimal import *

@ray.remote(num_gpus=1.0)
class SanityWorker(object):
    def __init__(self, train_set, test_set, train_transforms, worker_index, kwargs):
        self.device = torch.device("cuda")
        self.train_loader = Batches(Transform(train_set, train_transforms),
                        kwargs['batch_size'], shuffle=True,
                        set_random_choices=True, drop_last=True)
        self.test_loader = Batches(test_set, kwargs['batch_size'], shuffle=False,
                       drop_last=False)
        self.hp = kwargs
        self.worker_index = worker_index
        print(f"Initializing worker {self.worker_index}")
        self.opt_params = {
            "lr": kwargs['lr'],
            "momentum": kwargs['momentum'],
            "weight_decay": 5e-4*kwargs['batch_size'],
            "nesterov": kwargs['nesterov'],
            "dampening": 0,
        }
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        self.model = Net().cuda()
        torch.random.set_rng_state(rand_state)
        self.crit = nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.acc = Correct().to(self.device)
        step_number = 0
        self.opt = optim.SGD(self.model.parameters(), **self.param_values(step_number))
        # self._sketcher_init(**{'k': self.hp['k'], 'p2': self.hp['p2'], 'numCols': self.hp['numCols'], 'numRows': self.hp['numRows'], 'numBlocks': self.hp['numBlocks']})

    def param_values(self, step_number):
        #import pdb; pdb.set_trace()
        return {k: v(step_number) if callable(v) else v for k,v in self.opt_params.items()}
    
    def fetch_opt_params(self):
        assert len(self.opt.param_groups) == 1
        return {'lr': self.opt.param_groups[0]['lr'], 'momentum': self.opt.param_groups[0]['momentum'], 'weight_decay': self.opt.param_groups[0]['weight_decay'], 'nesterov': self.opt.param_groups[0]['nesterov'], 'dampening': self.opt.param_groups[0]['dampening']}

    def train_epoch(self, step_number, training):
        model = self.model
        optimizer = self.opt
        if training:
            dataloader = self.train_loader
            opt_copy = copy.deepcopy(self.opt.param_groups)
        else:
            dataloader = self.test_loader
        criterion = self.crit
        accuracy = self.acc
        running_loss = 0.0
        running_acc = 0.0
        for idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = batch["input"]
            targets = batch["target"]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = accuracy(outputs, targets)
            if training:
                step_number += 1
                optimizer.param_groups[0].update(**self.param_values(step_number))
                loss.sum().backward()
                optimizer.step()
            running_loss += loss.float().mean().detach().cpu().numpy()
            running_acc += acc.float().mean().detach().cpu().numpy()
        if training:
            return (running_loss/len(dataloader)), (running_acc/len(dataloader)), step_number, self.model_diff(opt_copy).cpu()
        else:
            return (running_loss/len(dataloader)), (running_acc/len(dataloader))

    def model_diff(self, opt_copy):
        diff_vec = []
        for group_id, param_group in enumerate(self.opt.param_groups):
            for idx, p in enumerate(param_group['params']):
                # calculate the difference between the current model and the stored model
                diff_vec.append(opt_copy[group_id]['params'][idx].data.view(-1).float() - p.data.view(-1).float())
                # reset the current model to the stored model for later
                p.data = opt_copy[group_id]['params'][idx].data
        self.diff_vec = torch.cat(diff_vec).to(self.device)
        #import pdb; pdb.set_trace()
        #print(f"Found a difference of {torch.sum(self.diff_vec)}")
        # return self.diff_vec
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

    def apply_update(self, weight_update):
        weight_update = weight_update.to(self.device)
        start = 0
        for param_group in self.opt.param_groups:
            for p in param_group['params']:
                end = start + torch.numel(p)
                # we previously had diff_vec = copy - (copy - grad) = grad, so subtract here 
                p.data -= weight_update[start:end].reshape(p.data.shape)
                start = end

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
        self.sketch = CSVec(d=self.sketch_mask.sum().item(), c=numCols, r=numRows, 
                            device=self.device, nChunks=1, numBlocks=numBlocks)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sketched", action="store_true")
parser.add_argument("--sketch_biases", action="store_true")
parser.add_argument("--sketch_params_larger_than", action="store_true")
parser.add_argument("-k", type=int, default=50000)
parser.add_argument("--p2", type=int, default=1)
parser.add_argument("--cols", type=int, default=500000)
parser.add_argument("--rows", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_blocks", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=24)
parser.add_argument("--epochs_per_iter", type=int, default=1)
parser.add_argument("--optimizer", type=str, default="SGD")
parser.add_argument("--criterion", type=str, default="CrossEntropyLoss")
args = parser.parse_args()
#args.batch_size = math.ceil(args.batch_size/args.num_workers) * args.num_workers

print('Downloading datasets')
DATA_DIR = "sample_data"
dataset = cifar10(DATA_DIR)

lr_schedule = PiecewiseLinear([0, 5, args.epochs], [0, 0.4, 0])
train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
lr = lambda step: lr_schedule(step/len(test_train_batch))/args.batch_size
print('Starting timer')
timer = Timer()

print('Preprocessing training data')
train_set = list(zip(
        transpose(normalise(pad(dataset['train']['data'], 4))),
        dataset['train']['labels']))
train_sets = np.array_split(train_set, args.num_workers)
test_train_batch = Batches(Transform(train_sets[0], train_transforms),
                            args.batch_size, shuffle=True,
                                                set_random_choices=True, drop_last=True)
print('Finished in {:.2f} seconds'.format(timer()))
print('Preprocessing test data')
test_set = list(zip(transpose(normalise(dataset['test']['data'])),
                    dataset['test']['labels']))
def chunker_list(seq, size):
    return (seq[i::size] for i in range(size))
test_sets = list(chunker_list(test_set, args.num_workers))
print('Finished in {:.2f} seconds'.format(timer()))

TSV = TSVLogger()

kwargs = {
    "k": args.k,
    "p2": args.p2,
    "numCols": args.cols,
    "numRows": args.rows,
    "numBlocks": args.num_blocks,
    "lr": lr,
    "momentum": args.momentum,
    "optimizer" : args.optimizer,
    "criterion": args.criterion,
    "weight_decay": 5e-4*args.batch_size,
    "nesterov": args.nesterov,
    "dampening": 0,
    "metrics": ['loss', 'acc'],
    "batch_size": args.batch_size,
}
ray.init(num_gpus=8)
workers = [SanityWorker.remote(train_sets[worker_index], test_sets[worker_index], train_transforms, worker_index, kwargs) for worker_index in range(args.num_workers)]

def train_worker(workers, epochs=24, loggers=(), timer=None):
    timer = timer or Timer()
    step_number = 0
    for epoch in range(epochs):
        train_loss, train_acc, step_number, diff_vec = list(zip(*ray.get([worker.train_epoch.remote(step_number, True) for worker in workers])))
        step_number = step_number[0]
        diff_vec = torch.mean(torch.stack(diff_vec), dim=0)
        ray.wait([worker.apply_update.remote(diff_vec) for i,worker in enumerate(workers)])
        train_time = timer()
        test_loss, test_acc = list(zip(*ray.get([worker.train_epoch.remote(step_number, False) for worker in workers])))
        test_time = timer()
        stats = {
            'train_time': train_time,
            'train_loss': np.mean(np.stack([i for i in train_loss])),
            'train_acc': np.mean(np.stack([i for i in train_acc])),
            'test_time': test_time,
            'test_loss': np.mean(np.stack([i for i in test_loss])),
            'test_acc': np.mean(np.stack([i for i in test_acc])),
            'total_time': timer.total_time
        }
        param_values = ray.get(workers[0].fetch_opt_params.remote())
        lr = param_values['lr'] * args.batch_size
        momentum = param_values['momentum']
        weight_decay = param_values['weight_decay']
        nesterov = param_values['nesterov']
        dampening = param_values['dampening']
        #summary = union({'epoch': epoch+1, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay, 'nesterov': nesterov, 'dampening': dampening}, stats)
        #lr = param_values(step_number)['lr'] * args.batch_size
        summary = union({'epoch': epoch+1, 'lr': lr}, stats)
        for logger in loggers:
            logger.append(summary)
    return summary

train_worker(workers, args.epochs,
      loggers=(TableLogger(), TSV), timer=timer)
