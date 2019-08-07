import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ray.init(ignore_reinit_error=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CONSTANTS
epochs = 1
batch_size = 1
D_in, D_out, H_sizes = 2, 4, [2,4]

x = torch.randn(batch_size, D_in, device=device)
y = torch.randn(batch_size, D_out, device=device)
num_workers = 2

from sketched_classes import SketchedModel, SketchedWorker, SketchedLoss, SketchedOptimizer

class FCNet(nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes):
        super(FCNet, self).__init__()
        self.layers = nn.ModuleList()
        last_size = in_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(last_size, size))
            last_size = size
        self.final = nn.Linear(last_size, out_size)
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.final(x)

model_config = {
    "in_size": D_in,
    "out_size": D_out,
    "hidden_sizes": H_sizes,
}
model_cls = FCNet
sketched_params = [10, 1, 100, 5, 0, 1, 1e-3, 0.9, 0, 0, False]
workers = [SketchedWorker.remote(*sketched_params) for _ in range(2)]
sketched_model = SketchedModel(model_cls, model_config, workers)
opt = optim.SGD(sketched_model.parameters(), lr=1e-3)
sketched_opt = SketchedOptimizer(opt, workers)

batch_size = 32
x = torch.zeros(batch_size, D_in, device="cpu")
y = torch.ones(batch_size, D_out, device="cpu")
criterion = torch.nn.MSELoss(reduction='sum')
sketched_criterion = SketchedLoss(criterion, workers)
lambda1 = lambda epoch: (epoch + 1) // 30 
scheduler = optim.lr_scheduler.LambdaLR(sketched_opt, lr_lambda=[lambda1])

steps = 4
for i in range(steps):
    train_loss = sketched_criterion(sketched_model(x), y)
    print(train_loss.mean())
    train_loss.backward()
    sketched_opt.step()
    #if i % 1000 == 0:
    #    scheduler.step()
