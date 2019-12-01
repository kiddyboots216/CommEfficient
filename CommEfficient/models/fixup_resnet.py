import torch
import torch.nn as nn
import numpy as np
from fixup.imagenet.models.fixup_resnet_imagenet import FixupBottleneck

__all__ = ["FixupResNet50"]

class FixupResNet50:
    def __init__(self, **kwargs):
        super().__init__(FixupBottleneck, [3, 4, 6, 3], **kwargs)
