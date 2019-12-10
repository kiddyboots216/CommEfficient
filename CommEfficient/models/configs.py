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
        self.lr_scale = 0.08
        # Override lr schedule set by ResNet9Config
        self.set_lr_schedule()

class FixupResNet50Config(ModelConfig):
    def __init__(self):
        super().__init__()
        self.model_config = {}
        """
        self.lr_schedule = PiecewiseLinear(
                [0,   7,   13,   13,     22,      25,       25,     28],
                [1.0, 2.0, 0.25, 0.4375, 0.04375, 0.004375, 0.0025, 2.5e-4]
            )
        """
        self.lr_schedule = PiecewiseLinear(
                [0,   9,   17,   17,     29,      33,       33,     37],
                [1.0, 2.0, 0.25, 0.4375, 0.04375, 0.004375, 0.0025, 2.5e-4]
            )

        """
        [{'ep': 0, 'sz': 128, 'bs': 512, 'trndir': '-sz/160'},
         {'ep': (0, 7), 'lr': (1.0, 2.0)},
         {'ep': (7, 13), 'lr': (2.0, 0.25)},
         {'ep': 13, 'sz': 224, 'bs': 224, 'trndir': '-sz/320', 'min_scale': 0.087},
         {'ep': (13, 22), 'lr': (0.4375, 0.04375)},
         {'ep': (22, 25), 'lr': (0.04375, 0.004375)},
         {'ep': 25, 'sz': 288, 'bs': 128, 'min_scale': 0.5, 'rect_val': True},
         {'ep': (25, 28), 'lr': (0.0025, 0.00025)}]
         """
