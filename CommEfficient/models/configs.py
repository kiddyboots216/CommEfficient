class ModelConfig:
    def __init__(self):
        pass

    def set_args(self, args):
        for name, val in self.__dict__.items():
            setattr(args, name, val)

class FixupResNet50Config(ModelConfig):
    def __init__(self):
        super().__init__()
        self.model_config = {}
        self.lr_scale = 0.1
        #self.lr_schedule = lambda epoch: 0.1 * 0.1 ** (epoch // 30)
        self.lr_schedule = PiecewiseLinear(
                [0, 30, 30, 60, 60, 90, 90, 100],
                [0.1, 0.1, 0.01, 0.01, 0.001, 0.001, 0.0001, 0.0001]
            )
        """
        self.lr_schedule = PiecewiseLinear(
                [0,   7,   13,   13,     22,      25,       25,     28],
                [1.0, 2.0, 0.25, 0.4375, 0.04375, 0.004375, 0.0025, 2.5e-4]
            )
        self.lr_schedule = PiecewiseLinear(
                [0,   9,   17,   17,     29,      33,       33,     37],
                [1.0, 2.0, 0.25, 0.4375, 0.04375, 0.004375, 0.0025, 2.5e-4]
            )

        [{'ep': 0, 'sz': 128, 'bs': 512, 'trndir': '-sz/160'},
         {'ep': (0, 7), 'lr': (1.0, 2.0)},
         {'ep': (7, 13), 'lr': (2.0, 0.25)},
         {'ep': 13, 'sz': 224, 'bs': 224, 'trndir': '-sz/320', 'min_scale': 0.087},
         {'ep': (13, 22), 'lr': (0.4375, 0.04375)},
         {'ep': (22, 25), 'lr': (0.04375, 0.004375)},
         {'ep': 25, 'sz': 288, 'bs': 128, 'min_scale': 0.5, 'rect_val': True},
         {'ep': (25, 28), 'lr': (0.0025, 0.00025)}]
         """
