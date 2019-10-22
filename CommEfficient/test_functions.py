from dp_functions import DPGaussianOptimizer
import torch
from functions import FedCommEffModel, FedCommEffOptimizer, \
        FedCommEffCriterion, FedCommEffMetric
from utils import parse_args

import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = parse_args(default_lr=0.4)
    D_in, D_out, H_sizes = 2, 4, [2,4]
    n_clients = 2
    epochs, batch_size = 10, 1
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
    opt = FedCommEffOptimizer(opt, args)

    xs = torch.randn(batch_size, D_in)
    ys = torch.randn(batch_size, D_out)
    minibatch = [xs, ys]
    minibatches = [minibatch for _ in range(n_clients)]
    idx = [i for i in range(n_clients)]
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, 
            lambda x: x)
    for _ in range(epochs):
        model.train(True)
        loss, acc = model(minibatches, idx)
        print(loss)
        print(acc)
        opt.step(idx)
        scheduler.step()
        model.train(False)
        loss, acc = model(minibatch, idx)
        print(loss)
        print(acc)
