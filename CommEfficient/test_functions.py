from dp_functions import DPGaussianOptimizer
import torch
from functions import FedCommEffModel, FedCommEffOptimizer, \
        FedCommEffCriterion, FedCommEffMetric
from utils import parse_args
from minimal import Correct

import multiprocessing
class FCNet(torch.nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes):
        super(FCNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        last_size = in_size
        for size in hidden_sizes:
            self.layers.append(torch.nn.Linear(last_size, size))
            last_size = size
        self.final = torch.nn.Linear(last_size, out_size)
    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.relu(layer(x))
        return self.final(x)
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = parse_args(default_lr=0.4)
    args.is_supervised = True
    args.mode = "true_topk"
    args.momentum_type = "virtual"
    args.error_type = "virtual"
    args.k = 10
    args.num_results_train = 2
    args.num_results_val = 2
    args.epochs = 2
    args.batch_size = 2
    D_in, D_out, H_sizes = 2, 4, [2,4]
    model_config = {
        "in_size": D_in,
        "out_size": D_out,
        "hidden_sizes": H_sizes,
    }
    model_cls = FCNet
    # instantiate ALL the things
    model = model_cls(**model_config)
    opt = torch.optim.SGD(model.parameters(), lr=1)
    # even for median or mean, each worker still sums gradients locally
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    accuracy = Correct()
    # FedComm-ify everything
    criterion = FedCommEffCriterion(criterion, args)
    accuracy = FedCommEffMetric(accuracy, args)
    model = FedCommEffModel(model, args)
    #opt = FedCommEffOptimizer(opt, args)
    opt = DPGaussianOptimizer(args, opt)

    xs = torch.randn(args.batch_size, D_in)
    ys = torch.ones(args.batch_size).long()
    minibatch = [xs, ys]
    minibatches = [minibatch for _ in range(args.num_workers)]
    idx = [i for i in range(args.num_workers)]
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, 
            lambda x: x)
    for _ in range(args.epochs):
        model.train(True)
        loss, acc = model(minibatches, idx)
        print(loss)
        print(acc)
        opt.step(idx)
        scheduler.step()
        model.train(False)
        loss, acc = model(minibatches, idx)
        print(loss)
        print(acc)
        print(opt.ledger.get_formatted_ledger())
