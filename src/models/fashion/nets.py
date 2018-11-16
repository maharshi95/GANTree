# # from __future__ import print_function

import torch as tr
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from base.hyperparams import Hyperparams
from exp_context import ExperimentContext

from modules.commons import NLinear, ConvBlock
from base.model import BaseModel

H = ExperimentContext.Hyperparams  # type: Hyperparams


class UpConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, bn=True, kernel_size=5, stride=2, output_padding=(0, 0), padding=(2, 2)):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                     output_padding=output_padding)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        super(UpConvBlock, self).__init__(*layers)


class ImgEncoder(BaseModel):
    def __init__(self, z_size, out_scale=4.0):
        super(ImgEncoder, self).__init__()

        self._np_out_scale = out_scale
        self.z_size = z_size

        self.out_scale = Parameter(tr.tensor(out_scale), requires_grad=False)

        # 1  * 28 x 28
        self.conv1 = ConvBlock(1, 32, kernel_size=5, stride=1, padding='same')
        # 32 * 28 x 28
        self.conv2 = ConvBlock(32, 64, kernel_size=5, stride=2, padding='same')
        # 64 * 14 x 14
        self.conv3 = ConvBlock(64, 128, kernel_size=5, stride=2, padding='same')
        # 128 * 7 x 7
        self.conv4 = ConvBlock(128, 256, kernel_size=5, stride=2, padding='same')
        # 256 * 4 x 4
        self.fc1 = nn.Linear(256 * (4 * 4), 200)
        # 200
        self.fc2 = nn.Linear(200, z_size)
        # z_size

        self.init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        z = self.out_scale * tr.tanh(x / self.out_scale)
        return z

    def copy(self, *args, **kwargs):
        return super(ImgEncoder, self).copy(self.z_size,out_scale=self._np_out_scale)

    @property
    def z_bounds(self):
        return self._np_out_scale


class ImgDecoder(BaseModel):
    def __init__(self, z_size, out_scale=4.0):
        """
        Using Padding Formula output = (input - 1) * stride - 2 * padding + kernel + output_padding
        """
        self.z_size = z_size
        super(ImgDecoder, self).__init__()

        self._np_out_scale = out_scale
        self.out_scale = Parameter(tr.tensor(out_scale), requires_grad=False)

        self.fc1 = nn.Linear(z_size, 200)
        self.fc2 = nn.Linear(200, 4 * 4 * 128)

        # 128 * 4 * 4
        self.tconv1 = UpConvBlock(128, 64, kernel_size=5, stride=2)
        # 64 * 7 * 7
        self.tconv2 = UpConvBlock(64, 32, kernel_size=5, stride=2, output_padding=(1, 1))
        # 32 * 14 * 14
        self.tconv3 = UpConvBlock(32, 16, kernel_size=5, stride=1)
        # 16 * 14 * 14
        self.tconv4 = UpConvBlock(16, 1, kernel_size=5, stride=2, output_padding=(1, 1))
        # 1 * 28 * 28
        self.init_params()

    @property
    def z_bounds(self):
        return self._np_out_scale

    def forward(self, z):
        z = self.fc1(z)
        z = F.leaky_relu(z)

        z = self.fc2(z)
        z = z.view(z.shape[0], 128, 4, 4)

        z = self.tconv1(z)
        z = self.tconv2(z)
        z = self.tconv3(z)
        z = self.tconv4(z)
        z = tr.tanh(z)
        return z

    def copy(self, *args, **kwargs):
        return super(ImgDecoder, self).copy(self.z_size,out_scale=self._np_out_scale)


class ImgDiscx(BaseModel):
    def __init__(self, n_batch_logits):
        super(ImgDiscx, self).__init__()

        self.n_batch_logits = n_batch_logits

        # 1  * 28 x 28
        self.conv1 = ConvBlock(1, 32, kernel_size=5, stride=1, padding='same')
        # 32 * 28 x 28
        self.conv2 = ConvBlock(32, 64, kernel_size=5, stride=2, padding='same')
        # 64 * 14 x 14
        self.conv3 = ConvBlock(64, 128, kernel_size=5, stride=2, padding='same')
        # 128 * 7 x 7
        self.conv4 = ConvBlock(128, 256, kernel_size=5, stride=2, padding='same')
        # 256 * 4 x 4
        self.fc1 = nn.Linear(256 * (4 * 4), 100)

        self.sample_logit_block = NLinear(100, [32, 32, 1])

        self.fc2 = nn.Linear(100, 32)

        self.batch_logit_block = NLinear(32 * n_batch_logits, [32, 32, 1])

        self.init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        common_layer = self.fc1(x)

        sample_logits = self.sample_logit_block(common_layer)

        batch_units_branch = self.fc2(common_layer)
        batch_units_layer = batch_units_branch.view(-1, self.n_batch_logits * 32)

        batch_logits = self.batch_logit_block(batch_units_layer)

        return sample_logits, batch_logits

    def discriminate(self, x):
        with tr.no_grad():
            sample_logits, _ = self.forward(x)
            preds = sample_logits >= 0.
            return preds[:, 0]

    def copy(self, *args, **kwargs):
        return super(ImgDiscx, self).copy(n_batch_logits=self.n_batch_logits)


class ImgDiscz(BaseModel):
    def __init__(self, n_batch_logits):
        super(ImgDiscz, self).__init__()
        self._np_n_batch_logits = n_batch_logits

        self._np_n_batch_logits = n_batch_logits
        self.n_common_features = 32

        self.n_batch_logits = Parameter(tr.tensor(n_batch_logits), requires_grad=False)

        self.common_block = NLinear(H.z_size, [16, 32, 64, 128, self.n_common_features], act=nn.ELU)

        self.sample_logit_block = NLinear(self.n_common_features, [32, 64, 32, 16, 1])
        self.batch_logit_block = NLinear(self.n_common_features * n_batch_logits, [32, 64, 32, 16, 1])

        self.init_params()

    def forward(self, z):
        inter = self.common_block(z)
        inter_view = inter.view(-1, self.n_common_features * self.n_batch_logits)

        sample_logits = self.sample_logit_block(inter)
        batch_logits = self.batch_logit_block(inter_view)

        return sample_logits, batch_logits

    def discriminate(self, z):
        with tr.no_grad():
            sample_logits, _ = self.forward(z)
            preds = sample_logits >= 0.
            return preds[:, 0]

    def copy(self, *args, **kwargs):
        return super(ImgDiscz, self).copy(n_batch_logits=self.n_batch_logits)
