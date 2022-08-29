from torch import nn


class MeanPooling(nn.Module):
    def __init__(self, dim: int = 1):
        super(MeanPooling, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)


class ClsPooling(nn.Module):
    def __init__(self, cls_position: int = 0):
        super(ClsPooling, self).__init__()
        self.cls_position = cls_position

    def forward(self, x):
        return x[:, self.cls_position]
