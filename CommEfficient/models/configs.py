from collections import namedtuple
import numpy as np

__all__ = ["FixupResNet9Config", "ResNet9Config"]

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

class ModelConfig:
    def __init__(self):
        pass

    def set_args(self, args):
        for name, val in self.__dict__.items():
            setattr(args, name, val)

class ResNet9Config(ModelConfig):
    def __init__(self):
        self.model_config = {
                'channels': {'prep': 64, 'layer1': 128,
                'layer2': 256, 'layer3': 512},
        }
        self.lr_scale = 0.4
        self.batch_size = 512
        self.weight_decay = 5e-4
        self.set_lr_schedule()

    def set_lr_schedule(self):
        self.lr_schedule = PiecewiseLinear([0, 5, 24],
                                  [0, self.lr_scale, 0])

class FixupResNet9Config(ResNet9Config):
    def __init__(self):
        super().__init__()
        self.lr_scale = 0.06
        # Override lr schedule set by ResNet9Config
        self.set_lr_schedule()
