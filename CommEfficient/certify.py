import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models import LinearModel
import argparse
def get_data(name, augment=False, **kwargs):
    if name == "cifar10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.CIFAR10(root=".data", train=True,
                                     transform=transforms.Compose(train_transforms),
                                     download=True)

        test_set = datasets.CIFAR10(root=".data", train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), normalize]
                                    ))
    return train_set, test_set

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument("--bound", type=float)
args = parser.parse_args()
feature_path = "/data/nvme/ashwinee/datasets/CIFAR10Pretrained/features/cifar100_resnext"
x_test = np.load(f"{feature_path}_test.npy")
train_data, test_data = get_data("cifar10", augment=False)
y_test = np.asarray(test_data.targets)
testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
BN_PATH = "bn_stats.pt"
bn_stats = torch.load(BN_PATH, map_location=torch.device(device))
n_features = x_test.shape[-1]
model = LinearModel().to(device)
PATH = "handcrafted_dp_clipped_pretrained.pt"
model.load_state_dict(torch.load(PATH))
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
diffs = np.zeros(10000)
for idx, data_tuple in enumerate(test_loader):
    data, target = data_tuple
    output = model(data.to(device))
    sorted_logits = output.sort()[0][0]
    highest_logits = sorted_logits[-2:]
    diff = abs(highest_logits[0] - highest_logits[1])
    diffs[idx] = diff

with open('logit_diffs.npy', 'wb') as f:
    np.save(f, diffs)

diffs = np.load("logit_diffs.npy")
diffs.sort()
print("Certified points:", 10000-np.searchsorted(diffs, args.bound))
