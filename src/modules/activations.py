import torch as tr
from torch import nn


class ScaledTanh(nn.Module):

    def __init__(self, scale=1.0):
        super(ScaledTanh, self).__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * tr.tanh(x / self.scale)


class CircularTanH(nn.Module):

    def __init__(self):
        super(CircularTanH, self).__init__()

    def forward(self, v):
        r = tr.norm(v, dim=-1)
        r_ = tr.tanh(r)
        return (r_ / r).unsqueeze(-1) * v


class ScaledCircularTanh(nn.Sequential):
    def __init__(self, scale=1.0):
        super(ScaledCircularTanh, self).__init__(*[
            ScaledTanh(scale),
            CircularTanH()
        ])
