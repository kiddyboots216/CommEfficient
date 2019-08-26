from functions import FedCommEffModel, FedCommEffOptimizer, FedCommEffLoss
import torch
import ray

if __name__ == "__main__":
    ray.init(redis_password='functional')
    D_in, D_out, H_sizes = 2, 4, [2,4]
    n_clients = 1
    epochs, batch_size = 4, 1
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
        'sketch': True,
        'num_cols': 1,
        'num_rows': 1,
        #'device': 'cpu',
        'device': 'cuda',
    }
    model_cls = FCNet
    xs = torch.randn(batch_size, D_in)
    ys = torch.randn(batch_size, D_out)
    batch = [xs, ys]
    batches = [batch for _ in range(n_clients)]
    idx = [i for i in range(n_clients)]
    comm_model = FedCommEffModel(model_cls, model_config, params)
    optimizer = torch.optim.SGD(comm_model.parameters(), lr=1)
    comm_optimizer = FedCommEffOptimizer(optimizer, params)
    criterion = torch.nn.MSELoss()
    comm_criterion = FedCommEffLoss(criterion, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(comm_optimizer, 
            lambda x: x)
    for _ in range(epochs):
        comm_model.train(True)
        grads = comm_model(batches, idx)
        comm_optimizer.step(grads, idx)
        scheduler.step()
        comm_model.train(False)
        outs = comm_model(batch, idx)
        print(ray.get(outs))
