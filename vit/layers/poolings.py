import torch.nn as nn


class MeanPooler(nn.Module):
    def __init__(self, dim: int = 1):
        super(MeanPooler, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)


class ClsPooler(nn.Module):
    def __init__(self, cls_position: int = 0):
        super(ClsPooler, self).__init__()
        self.cls_position = cls_position

    def forward(self, x):
        return x[:, self.cls_position]


__all__ = [
    "MeanPooler",
    "ClsPooler"
]
