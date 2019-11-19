from fed_aggregator import FedModel, FedOptimizer, FedCriterion, FedAccuracy
import torch
import ray

if __name__ == "__main__":
    ray.init(redis_password='functional')
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
    params = {
        'n_clients': n_clients,
        'p2': 1,
        'k': 1,
        'sketch_down': False,
        'topk_down': True,
        'sketch': True,
        'momentum_sketch': False,
        'virtual_momentum': True,
        'momentum': 0.9,
        'weight_decay': 1.0,
        'n_clients_per_round': 1,
        'num_cols': 1,
        'num_rows': 1,
        'num_blocks': 1,
        #'device': 'cpu',
        'device': 'cuda',
    }
    model_cls = FCNet
    xs = torch.randn(batch_size, D_in)
    ys = torch.randn(batch_size, D_out)
    minibatch = [xs, ys]
    minibatches = [minibatch for _ in range(n_clients)]
    idx = [i for i in range(n_clients)]
    model = FedModel(model_cls, model_config, params)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    opt = FedOptimizer(optimizer, params)
    criterion = torch.nn.MSELoss()
    comm_criterion = FedCriterion(criterion, params)
    fake_acc = FedAccuracy(criterion, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, 
            lambda x: x)
    for _ in range(epochs):
        model.train(True)
        outs, loss, acc, grads = model(minibatches, idx)
        opt.step(grads, idx)
        scheduler.step()
        # TODO: Fix train acc calculation
        batch_loss = ray.get(loss)
        batch_acc = ray.get(acc)
        print(batch_loss)
        print(batch_acc)
        model.train(False)
        outs, loss, acc = model(minibatch, idx)
        batch_loss = ray.get(loss)
        batch_acc = ray.get(acc)
        print(batch_loss)
        print(batch_acc)

