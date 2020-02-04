import torch.nn as nn
import torch.nn.functional as F
import models

__all__ = ["ResNet101LN"]

class ResNet101LN(nn.Module):
    def __init__(self, *args, num_classes=62, **kwargs):
        super().__init__()
        self.model = models.resnet101(num_classes=num_classes, norm_layer=nn.LayerNorm)

    def forward(self, x):
        return self.model(x)
