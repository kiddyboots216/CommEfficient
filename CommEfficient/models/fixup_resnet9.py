import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fixup.cifar.models.fixup_resnet_cifar import FixupBasicBlock, conv3x3

__all__ = ["FixupResNet9"]

class FixupLayer(nn.Module):
    """ conv, bias, relu, pool, followed by num_blocks FixupBasicBlocks """
    def __init__(self, in_channels, out_channels, num_blocks, pool):
        super().__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))
        self.pool = pool
        self.blocks = nn.Sequential(
                    *[FixupBasicBlock(out_channels, out_channels)
                      for _ in range(num_blocks)]
                )

    def forward(self, x):
        out = self.conv(x + self.bias1a) * self.scale + self.bias1b
        out = F.relu(out)
        if self.pool is not None:
            out = self.pool(out)
        for block in self.blocks:
            out = block(out)
        return out

class FixupResNet9(nn.Module):
    def __init__(self, channels=None, pool=nn.MaxPool2d(2)):
        super().__init__()
        self.num_layers = 2
        self.channels = channels or {"prep": 64, "layer1": 128,
                                     "layer2": 256, "layer3": 512}
        self.conv1 = conv3x3(3, self.channels["prep"])
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))

        self.layer1 = FixupLayer(self.channels["prep"],
                                 self.channels["layer1"],
                                 1, pool)
        self.layer2 = FixupLayer(self.channels["layer1"],
                                 self.channels["layer2"],
                                 0, pool)
        self.layer3 = FixupLayer(self.channels["layer2"],
                                 self.channels["layer3"],
                                 1, pool)

        self.pool = nn.MaxPool2d(4)
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(self.channels["layer3"], 10)

        # initialize conv1
        std = np.sqrt(2 /
                    (self.conv1.weight.shape[0]
                     * np.prod(self.conv1.weight.shape[2:]))
              )
        nn.init.normal_(self.conv1.weight, mean=0, std=std)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                std = np.sqrt(2 /
                            (m.conv1.weight.shape[0]
                             * np.prod(m.conv1.weight.shape[2:]))
                      ) * self.num_layers ** (-0.5)
                nn.init.normal_(m.conv1.weight, mean=0, std=std)
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, FixupLayer):
                std = np.sqrt(2 /
                            (m.conv.weight.shape[0]
                             * np.prod(m.conv.weight.shape[2:]))
                      )
                nn.init.normal_(m.conv.weight, mean=0, std=std)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x + self.bias1a) * self.scale + self.bias1b
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).view(out.size()[0], -1)
        out = self.linear(out + self.bias2)
        return out
