from collections import namedtuple

import torch as tr
from torch import nn
from torch.nn import functional as F

from configs import Config
from utils.tr_utils import ellipse_params, rotate


class NLinear(nn.Sequential):
    def __init__(self, in_feat, units, act=nn.ELU):
        layers = [nn.Linear(in_feat, units[0])]
        for i in range(len(units) - 1):
            in_feat, out_feat = units[i:i + 2]
            layers.append(act())
            layers.append(nn.Linear(in_feat, out_feat))

        super(NLinear, self).__init__(*layers)
        if Config.use_gpu:
            self.cuda()


ZParams = namedtuple('ZParams', 'means cov')


class SingleZTransform(nn.Module):
    def __init__(self, params):
        super(SingleZTransform, self).__init__()

        self.means, self.cov = params
        self.th, self.a, self.b = ellipse_params(self.cov)
        self.scale = tr.tensor([self.a, self.b], dtype=tr.float32)

    @property
    def params(self):
        return ZParams(self.means, self.cov)

    @params.setter
    def params(self, value):
        means, cov = value
        self.means, self.cov = tr.Tensor(means), tr.Tensor(cov)

    @property
    def inv_params(self):
        return ZParams(-self.means, self.cov)

    def normalize(self, x):
        x = x - self.means
        x = rotate(x, - self.th)
        x = x / self.scale
        return x

    def denormalize(self, x):
        x = x * self.scale
        x = rotate(x, self.th)
        x = x + self.means
        return x


class ZTransform(nn.Module):
    def __init__(self, src_params, target_params=None):
        super(ZTransform, self).__init__()
        # print(type(src_params))
        # print(src_params)
        src_params = list(map(tr.tensor, src_params))
        # print(src_params)
        # print(type(src_params))
        if target_params is None:
            target_params = tr.zeros(src_params[0].shape), tr.eye(src_params[0].shape[0])
        self.src_transform = SingleZTransform(src_params)
        self.target_transform = SingleZTransform(target_params)

    @property
    def src_params(self):
        return self.src_transform.params

    @property
    def target_params(self):
        return self.target_transform.params

    @src_params.setter
    def src_params(self, value):
        self.src_transform.params = value

    @target_params.setter
    def target_params(self, value):
        self.target_transform.params = value

    def forward(self, x):
        x = self.src_transform.normalize(x)
        x = self.target_transform.denormalize(x)
        return x

    def inv(self, x):
        x = self.target_transform.normalize(x)
        x = self.src_transform.denormalize(x)
        return x


#     use only when h == w for an image
def get_padding(stride, in_dim, kernel_dim, out_dim=None, mode='SAME'):
    k = kernel_dim
    if out_dim == None:
        out_dim = (in_dim + stride - 1) // stride
    if mode.lower() == 'same':
        val = stride * (out_dim - 1) - in_dim + k
        if val % 2 == 0:
            p1, p2 = val // 2, val // 2
        else:
            p1, p2 = (val + 1) // 2, (val + 1) // 2
        return (p1, p2, p1, p2)


class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, bn=True, kernel_size=3, stride=2, padding=None):
        super(ConvBlock, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel = kernel_size

        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride)]
        if bn:
            layers.append(nn.BatchNorm2d(out_filters, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        padding = self.padding
        if isinstance(padding, str):
            padding = get_padding(self.stride, input.shape[2], self.kernel, mode=self.padding)
        elif padding is None:
            padding = 0
        input = F.pad(input, padding)
        input = self.conv_block(input)
        return input
