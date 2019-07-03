import torch
import torch.nn as nn
import torch.optim as optim

from minimal import *

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
}
param_values = {
	"lr": lr,
    "momentum": args.momentum,
    "weight_decay": 5e-4*args.batch_size,
    "nesterov": args.nesterov,
    "dampening": 0,
}
model = Net().cuda()
criterion = nn.CrossEntropyLoss(reduction='none').cuda()
accuracy = Correct().cuda()
step_number = 0

def param_values(self):
    #import pdb; pdb.set_trace()
    return {k: v(step_number) if callable(v) else v for k,v in param_values.items()}

optimizer = optim.SGD(model.parameters(), **param_values())

# Create workers.

# track_dir = "sample_data"
# with track.trial(track_dir, None, param_map=vars(optim_args)):

train(model, optimizer, train_batches, test_batches, args.epochs,
      loggers=(TableLogger(), TSV), timer=timer)

def train_epoch(model, optimizer, dataloader, training):
	running_loss = 0.0
	running_acc = 0.0
	for idx, batch in enumerate(dataloader):
		inputs = batch["input"]
		targets = batch["target"]
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		acc = accuracy(outputs, targets)
		if training:
			optimizer.zero_grad()
			loss.sum().backward()
			optimizer.step()
		running_loss += loss.mean()
		running_acc += acc.mean()
	return (running_loss/len(dataloader)).detach().cpu().numpy(), (running_acc/len(dataloader)).detach().cpu().numpy()

def train(model, optimizer, train_loader, val_loader, 
			epochs=24, loggers=(), timer=None):
	timer = timer or Timer()
	for epoch in range(epochs):
		train_loss, train_acc = train_epoch(model, optimizer, train_loader, True)
		train_time = timer()
		test_loss, test_acc = train_epoch(model, optimizer, val_loader, False)
		test_time = timer()
		stats = {
			'train_time': train_time,
			'train_loss': train_loss,
			'train_acc': train_acc,
			'test_time': test_time,
			'test_loss': test_loss,
			'test_acc': test_acc,
			'total_time': timer.total_time
		}
		lr = param_values['lr'] * batch_size
		summary = union({'epoch': epoch+1, 'lr': lr}, stats)
		for logger in loggers:
			logger.append(summary)
	return summary

