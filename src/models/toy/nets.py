from __future__ import print_function

import torch as tr
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

from base.hyperparams import Hyperparams
from exp_context import ExperimentContext
from modules.activations import CircularTanH

from modules.commons import NLinear
from base.model import BaseModel

H = ExperimentContext.Hyperparams  # type: Hyperparams


class ToyEncoder(BaseModel):
    def __init__(self, in_feat=2, out_feat=2, out_scale=4.0):
        super(ToyEncoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self._np_out_scale = out_scale

        n_units = [32, 64, 128, 64, 32, out_feat]

        self.out_scale = Parameter(tr.tensor(out_scale), requires_grad=False)

        self.linear = NLinear(in_feat, n_units, act=nn.ELU)
        self.act = CircularTanH() if H.circular_bounds else nn.Tanh()

        self.init_params()

    @property
    def z_bounds(self):
        return self._np_out_scale

    def forward(self, x):
        x = self.linear(x)
        return self.out_scale * self.act(x / self.out_scale)

    def copy(self, *args, **kwargs):
        return super(ToyEncoder, self).copy(in_feat=self.in_feat,
                                            out_feat=self.out_feat,
                                            out_scale=self._np_out_scale)


class ToyDecoder(BaseModel):
    def __init__(self, in_feat=2, out_feat=2, out_scale=4.0):
        super(ToyDecoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self._np_out_scale = out_scale

        n_units = [32, 64, 128, 64, 32, out_feat]

        self.linear = NLinear(in_feat, n_units, act=nn.ELU)
        self.act = nn.Tanh()
        self.out_scale = Parameter(tr.tensor(out_scale), requires_grad=False)
        self.init_params()

    def forward(self, x):
        x = self.linear(x)
        return x
        # return self.out_scale * self.act(x / self.out_scale)

    def copy(self, *args, **kwargs):
        return super(ToyDecoder, self).copy(in_feat=self.in_feat,
                                            out_feat=self.out_feat,
                                            out_scale=self._np_out_scale)


class ToyDisc(BaseModel):
    def __init__(self, in_feat, n_batch_logits):
        super(ToyDisc, self).__init__()

        self.in_feat = in_feat
        self._np_n_batch_logits = n_batch_logits

        self.n_fused_features = 32
        self.n_common_features = 8
        self.n_batch_logits = Parameter(tr.tensor(n_batch_logits), requires_grad=False)

        self.common_block = NLinear(in_feat, [16, 32, 64, 128, self.n_common_features], act=nn.ELU)

        self.sample_logit_block = NLinear(self.n_common_features, [32, 1])
        self.batch_logit_block = NLinear(self.n_common_features * n_batch_logits, [16, 16, 1])

        self.init_params()

    def forward(self, x):
        inter = self.common_block(x)
        inter_view = inter.view(-1, self.n_common_features * self.n_batch_logits)

        sample_logits = self.sample_logit_block(inter)
        batch_logits = self.batch_logit_block(inter_view)

        return sample_logits, batch_logits

    def discriminate(self, x):
        with tr.no_grad():
            sample_logits, _ = self.forward(x)
            preds = sample_logits >= 0.
            return preds[:, 0]

    def copy(self, *args, **kwargs):
        return super(ToyDisc, self).copy(in_feat=self.in_feat,
                                         n_batch_logits=self._np_n_batch_logits)


class DualDisc(BaseModel):
    def __init__(self, x_in_feat=2, z_in_feat=2):
        super(DualDisc, self).__init__()

        self.n_fused_features = 32
        self.n_common_features = 8

        self.common_block_x = NLinear(x_in_feat, [16, self.n_fused_features], act=nn.ELU)
        self.common_block_z = NLinear(z_in_feat, [16, self.n_fused_features], act=nn.ELU)

        temp = self.n_fused_features if H.disc_type in {'x', 'z'} else self.n_fused_features * 2

        self.common_block_2 = NLinear(temp, [64, 128, self.n_common_features], act=nn.ELU)

        self.sample_logit_block = NLinear(self.n_common_features, [32, 1])
        self.batch_logit_block = NLinear(self.n_common_features * n_batch_logits, [16, 16, 1])

        self.init_params()

    def fused_features(self, x, z):
        if H.disc_type == 'x':
            x = self.common_block_x(x)
        elif H.disc_type == 'z':
            x = self.common_block_z(z)
        else:
            x = tr.cat([
                self.common_block_x(x),
                self.common_block_z(z)
            ], dim=-1)
        #
        x = F.elu(x)

        inter = self.common_block_2(x)
        inter = F.elu(inter)
        return inter

    def forward(self, x, z):
        inter = self.fused_features(x, z)
        inter_view = inter.view(-1, self.n_common_features * n_batch_logits)

        sample_logits = self.sample_logit_block(inter)
        batch_logits = self.batch_logit_block(inter_view)

        return sample_logits, batch_logits

    def discriminate(self, x, z):
        with tr.no_grad():
            inter = self.fused_features(x, z)
            sample_logits = self.sample_logit_block(inter)
            preds = sample_logits >= 0.
            return preds[:, 0]
