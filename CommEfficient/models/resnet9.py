import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResNet9"]

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x):
        return x*self.weight


def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False,
               bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False

    return m

#Network definition
class ConvBN(nn.Module):
    def __init__(self, iid, c_in, c_out, bn_weight_init=1.0, pool=None, **kw):
        super().__init__()
        self.pool = pool
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn = batch_norm(c_out, bn_weight_init=bn_weight_init, **kw)
        self.iid = iid
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.iid:
            out = self.relu(self.bn(self.conv(x)))
        else:
            out = self.relu(self.conv(x))
        if self.pool:
            out = self.pool(out)
        return out


class Residual(nn.Module):
    def __init__(self, iid, c, **kw):
        super().__init__()
        self.res1 = ConvBN(iid, c, c, **kw)
        self.res2 = ConvBN(iid, c, c, **kw)

    def forward(self, x):
        return x + F.relu(self.res2(self.res1(x)))

class BasicNet(nn.Module):
    def __init__(self, iid, channels, weight,  pool, **kw):
        super().__init__()
        self.prep = ConvBN(iid, 3, channels['prep'], **kw)

        self.layer1 = ConvBN(iid, channels['prep'], channels['layer1'],
                             pool=pool, **kw)
        self.res1 = Residual(iid, channels['layer1'], **kw)

        self.layer2 = ConvBN(iid, channels['layer1'], channels['layer2'],
                             pool=pool, **kw)

        self.layer3 = ConvBN(iid, channels['layer2'], channels['layer3'],
                             pool=pool, **kw)
        self.res3 = Residual(iid, channels['layer3'], **kw)

        self.pool = nn.MaxPool2d(4)
        self.linear = nn.Linear(channels['layer3'], 10, bias=False)
        self.classifier = Mul(weight)

    def forward(self, x):
        out = self.prep(x)
        out = self.res1(self.layer1(out))
        out = self.layer2(out)
        out = self.res3(self.layer3(out))

        out = self.pool(out).view(out.size()[0], -1)
        out = self.classifier(self.linear(out))
        return out

class ResNet9(nn.Module):
    def __init__(self, iid=True, channels=None, weight=0.125, pool=nn.MaxPool2d(2),
                 extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
        super().__init__()
        channels = channels or {'prep': 64, 'layer1': 128,
                                'layer2': 256, 'layer3': 512}
        print(f"Using BatchNorm: {iid}")
        self.n = BasicNet(iid, channels, weight, pool, **kw)
        #for layer in res_layers:
        #    n[layer]['residual'] = residual(channels[layer], **kw)
        #for layer in extra_layers:
        #    n[layer]['extra'] = ConvBN(channels[layer], channels[layer], **kw)
    def forward(self, x):
        return self.n(x)

