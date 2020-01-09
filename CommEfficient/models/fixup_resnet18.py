import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["ResNet18", "FixupResNet18"]

class Mul(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.scale

class Add(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.bias

class FixupBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #self.bias1a = nn.Parameter(torch.zeros(1))
        self.add1a = Add()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        #self.bias1b = nn.Parameter(torch.zeros(1))
        self.add1b = Add()
        #self.bias2a = nn.Parameter(torch.zeros(1))
        self.add2a = Add()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #self.scale = nn.Parameter(torch.ones(1))
        self.mul = Mul()
        #self.bias2b = nn.Parameter(torch.zeros(1))
        self.add2b = Add()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=1, stride=stride,
                                      bias=False)

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x

        """
        out = self.conv1(x + self.bias1a)
        out = F.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b
        """
        out = self.conv1(self.add1a(x))
        out = F.relu(self.add1b(out))
        out = self.conv2(self.add2a(out))
        out = self.add2b(self.mul(out))

        return F.relu(out + shortcut)

 
class FixupResNet18(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super().__init__()

        self.num_layers = sum(num_blocks)
        self.in_channels = 64

        self.prep = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                              bias=False)

        self.layers = nn.Sequential(
            self._make_layer(64, 64, num_blocks[0], stride=1),
            self._make_layer(64, 128, num_blocks[1], stride=2),
            self._make_layer(128, 256, num_blocks[2], stride=2),
            self._make_layer(256, 256, num_blocks[3], stride=2),
        )

        self.classifier = nn.Linear(512, num_classes)

        # initialize
        for m in self.modules():
            if isinstance(m, FixupBlock):
                std = np.sqrt(
                        2 / (m.conv1.weight.shape[0]
                             * np.prod(m.conv1.weight.shape[2:]))
                        ) * self.num_layers ** (-0.5)
                nn.init.normal_(m.conv1.weight, mean=0, std=std)

                nn.init.constant_(m.conv2.weight, 0)
                if hasattr(m, "shortcut"):
                    std = np.sqrt(
                            2 / (m.shortcut.weight.shape[0]
                                 * np.prod(m.shortcut.weight.shape[2:]))
                          )
                    nn.init.normal_(m.shortcut.weight, mean=0, std=std)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        std = np.sqrt(2 / (self.prep.weight.shape[0]
                           * np.prod(self.prep.weight.shape[2:])))
        nn.init.normal_(self.prep.weight, mean=0, std=std)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(FixupBlock(in_channels=in_channels,
                                     out_channels=out_channels,
                                     stride=stride))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.prep(x))

        x = self.layers(x)

        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)

        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)

        x = torch.cat([x_avg, x_max], dim=-1)

        x = self.classifier(x)

        return x


class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #self.bn1   = nn.BatchNorm2d(in_channels)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False)
            )

    def forward(self, x):
        #out = F.relu(self.bn1(x))
        #out = self.conv1(out)
        #out = self.conv2(F.relu(self.bn2(out)))
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x

        return out + shortcut


class ResNet18(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            #nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            self._make_layer(64, 64, num_blocks[0], stride=1),
            self._make_layer(64, 128, num_blocks[1], stride=2),
            self._make_layer(128, 256, num_blocks[2], stride=2),
            self._make_layer(256, 256, num_blocks[3], stride=2),
        )

        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        #x = x.half()
        x = self.prep(x)

        x = self.layers(x)

        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)

        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)

        x = torch.cat([x_avg, x_max], dim=-1)

        x = self.classifier(x)

        return x


