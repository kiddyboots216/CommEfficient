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
from . import ParameterServer
from . import Worker

if __name__ == "__main__":
	print('Downloading datasets')
	DATA_DIR = "sample_data"
	dataset = cifar10(DATA_DIR)

	epochs = 24
	lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
	batch_size = 512
	train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

	#model = Network(union(net(), losses)).to(device)
	model = Net().to(device)
	model = SketchedModel(model)

	# print('Warming up cudnn on random inputs')
	#for size in [batch_size, len(dataset['test']['labels']) % batch_size]:
	#    warmup_cudnn(model, size)

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
	lr = lambda step: lr_schedule(step/len(train_batches))/batch_size
	from easydict import EasyDict as edict
	sketched_args = edict({
	    "sketched": True,
	    "k": 50000,
	    "p2": 4,
	    "numCols": 500000,
	    "numRows": 5,
	    "numBlocks": 1,
	})
	weights = trainable_params(model)
	opt = SGD(weights, lr=lr, momentum=0.9,
	                           weight_decay=5e-4*batch_size,
	                           nesterov=True, dampening=0, **sketched_args)

	track_dir = "sample_data"
	with track.trial(track_dir, None, param_map=vars(sketched_args)):
	    train(model, opt, train_batches, test_batches, epochs,
	          loggers=(TableLogger(), TSV), timer=timer,
	          test_time_in_total=False)