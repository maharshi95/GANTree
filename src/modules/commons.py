from collections import namedtuple

import torch as tr
from torch import nn
from torch.nn import Parameter

from configs import Config
from utils import tr_utils
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
    def __init__(self, params, requires_grad=False, mean_grad=False, cov_grad=False):
        super(SingleZTransform, self).__init__()

        means = tr.tensor(params[0], dtype=tr.float32)
        cov = tr.tensor(params[1], dtype=tr.float32)
        self.means = Parameter(means, requires_grad=requires_grad or mean_grad)
        self.cov = Parameter(cov, requires_grad=requires_grad or cov_grad)

    @property
    def params(self):
        return ZParams(self.means, self.cov)

    @params.setter
    def params(self, value):
        means, cov = value
        self.means, self.cov = tr.Tensor(means), tr.Tensor(cov)

    @property
    def ellipse_params(self):
        return ellipse_params(self.cov)

    @property
    def inv_params(self):
        return ZParams(-self.means, self.cov)

    def normalize(self, x):
        return tr_utils.dist_normalize(x, self.means, self.cov)

    def denormalize(self, x):
        return tr_utils.dist_transform(x, self.means, self.cov)


class ZTransform(nn.Module):
    def __init__(self, src_params, target_params=None):
        super(ZTransform, self).__init__()
        src_params = map(tr.tensor, src_params)
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
