import torch
import torch.nn as nn

class SketchedModel:
    # not inheriting from nn.Module to avoid the fact that implementing
    # __getattr__ on a nn.Module is tricky, since self.model = model
    # doesn't actually add "model" to self.__dict__ -- instead, nn.Module
    # creates a key/value pair in some internal dictionary that keeps
    # track of submodules
    def __init__(self, model, sketchBiases=False, sketchParamsLargerThan=0):
        self.model = model
        # sketch everything larger than sketchParamsLargerThan
        for p in model.parameters():
            p.do_sketching = p.numel() >= sketchParamsLargerThan

        # override bias terms with whatever sketchBiases is
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                if m.bias is not None:
                    m.bias.do_sketching = sketchBiases

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        if name == "model":
            self.__dict__[name] = value
        else:
            self.model.setattr(name, value)
            
def topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    ret = torch.zeros_like(vec)

    # on a gpu, sorting is faster than pytorch's topk method
    topkIndices = torch.sort(vec**2)[1][-k:]
    #_, topkIndices = torch.topk(vec**2, k)

    ret[topkIndices] = vec[topkIndices]
    return ret