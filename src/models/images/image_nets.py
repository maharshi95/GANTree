# # from __future__ import print_function

import torch as tr
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from base.hyperparams import Hyperparams
from exp_context import ExperimentContext
from modules.activations import CircularTanH

from modules.commons import NLinear
from base.model import BaseModel

H = ExperimentContext.Hyperparams  # type: Hyperparams


class ConvBlock(nn.Sequential):
    def __init__(self, in_filters, out_filters, bn=True, kernel_size=3, stride=2):
        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride)]
        if bn:
            layers.append(nn.BatchNorm2d(out_filters, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        super(ConvBlock, self).__init__(*layers)


class UpConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, bn=True, kernel_size=5, stride=2, output_padding=(0, 0), padding=(2, 2)):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                     output_padding=output_padding)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        super(UpConvBlock, self).__init__(*layers)


class ImgEncoder(BaseModel):
    def __init__(self, out_scale=4.0):
        super(ImgEncoder, self).__init__()

        self._np_out_scale = out_scale
        self.out_scale = Parameter(tr.tensor(out_scale), requires_grad=False)

        self.conv1 = ConvBlock(H.input_channel, 32, kernel_size=5, stride=2)
        self.conv2 = ConvBlock(32, 64, kernel_size=5, stride=2)
        self.conv3 = ConvBlock(64, 128, kernel_size=5, stride=2)
        self.conv4 = ConvBlock(128, 256, kernel_size=5, stride=2)
        self.layer1 = nn.Linear(256 * (2 ** 2), 200)
        self.layer2 = nn.Linear(200, H.z_size)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.init_params()

    #     use only when h==w    for an image
    def padding(self, stride, in_dim, out_dim, kernel_dim, mode='SAME', ):
        s = stride
        k = kernel_dim
        if mode == 'SAME':
            val = s * (out_dim - 1) - in_dim + k
            return tuple([np.ceil(val / 2)] * 4)

    def forward(self, img):
        # print ('input_img', img.shape)

        img = F.pad(img, (2, 2, 2, 2))
        # print('in1', img.shape)

        img = F.pad(img, (2, 2, 2, 2))
        conv1 = self.conv1(img)
        # print(conv1.shape)

        conv1 = F.pad(conv1, (2, 2, 2, 2))
        conv2 = self.conv2(conv1)
        # print(conv2.shape)

        conv2 = F.pad(conv2, (2, 2, 2, 2))
        conv3 = self.conv3(conv2)
        # print(conv3.shape)

        conv3 = F.pad(conv3, (2, 2, 2, 2))
        out = self.conv4(conv3)
        # print (out.shape)

        out = out.view(out.size(0), -1)

        dense1 = self.layer1(out)
        dense1 = self.activation(dense1)
        dense2 = self.layer2(dense1)
        # print(dense2.shape)

        z = self.out_scale * F.tanh(dense2 / self.out_scale)

        return z

    def copy(self, *args, **kwargs):
        return super(ImgEncoder, self).copy(out_scale=self._np_out_scale)

    @property
    def z_bounds(self):
        return self._np_out_scale


class ImgDecoder(BaseModel):
    def __init__(self, out_scale=4.0):
        super(ImgDecoder, self).__init__()

        self._np_out_scale = out_scale
        self.out_scale = Parameter(tr.tensor(out_scale), requires_grad=False)

        self.linear1 = nn.Linear(H.z_size, 200)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.linear2 = nn.Linear(200, 4 * 4 * 32)

        self.tconv1 = UpConvBlock(in_channels=32, out_channels=16, kernel_size=5, stride=2, output_padding=(1, 1))
        self.tconv2 = UpConvBlock(in_channels=16, out_channels=8, kernel_size=5, stride=2, )
        self.tconv3 = UpConvBlock(in_channels=8, out_channels=1, kernel_size=4, stride=2, )

        self.init_params()

    @property
    def z_bounds(self):
        return self._np_out_scale

    def forward(self, z):
        # print(z.shape)
        fc1 = self.linear1(z)
        fc1 = self.activation(fc1)
        # print(fc1.shape)

        fc2 = self.linear2(fc1)
        # print(fc2.shape)

        fc2 = fc2.view(fc1.shape[0], 32, 4, 4)

        # print(fc2.shape)

        tconv1 = self.tconv1(fc2)
        # print(tconv1.shape)

        tconv2 = self.tconv2(tconv1)
        # print(tconv2.shape)

        tconv3 = self.tconv3(tconv2)
        # print(tconv3.shape)

        return tr.tanh(tconv3)
        # return  self.out_scale * F.tanh(tconv5 / self.out_scale)

    def copy(self, *args, **kwargs):
        return super(ImgDecoder, self).copy(out_scale=self._np_out_scale)


class ImgDiscx(BaseModel):

    def __init__(self, n_batch_logits):
        super(ImgDiscx, self).__init__()
        self._np_n_batch_logits = n_batch_logits
        self.n_batch_logits = Parameter(tr.tensor(n_batch_logits), requires_grad=False)

        self.conv1 = ConvBlock(H.input_channel, 32, kernel_size=5, stride=2)
        self.conv2 = ConvBlock(32, 64, kernel_size=5, stride=2)
        self.conv3 = ConvBlock(64, 128, kernel_size=5, stride=2)
        self.conv4 = ConvBlock(128, 256, kernel_size=5, stride=2)
        self.layer = nn.Linear(256 * (2 ** 2), 100)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.sample_logit_block = NLinear(100, [32, 32, 1])

        self.layer1 = nn.Linear(100, 32)
        self.batch_logit_block = NLinear(32 * n_batch_logits, [32, 32, 1])

        self.init_params()

    def forward(self, img):
        img = F.pad(img, (2, 2, 2, 2))
        # print('in1', img.shape)

        img = F.pad(img, (2, 2, 2, 2))
        conv1 = self.conv1(img)
        # print('conv1_out',conv1.shape)

        conv1 = F.pad(conv1, (2, 2, 2, 2))
        conv2 = self.conv2(conv1)
        # print('conv2_out',conv2.shape)

        conv2 = F.pad(conv2, (2, 2, 2, 2))
        conv3 = self.conv3(conv2)
        # print('conv3_out',conv3.shape)

        conv3 = F.pad(conv3, (2, 2, 2, 2))
        out = self.conv4(conv3)
        # print('conv4_out',out.shape)
        out = out.view(out.size(0), -1)
        # print (out.shape)

        z_out = self.layer(out)
        z_out = self.activation(z_out)
        # print('z_out_shape', z_out.shape)

        sample_logits = self.sample_logit_block(z_out)

        batch_logits_branch_layer1 = self.layer1(z_out)
        batch_in = batch_logits_branch_layer1.view(-1, 32 * self.n_batch_logits)
        batch_logits = self.batch_logit_block(batch_in)

        return sample_logits, batch_logits

    def discriminate(self, x):
        with tr.no_grad():
            sample_logits, _ = self.forward(x)
            preds = sample_logits >= 0.
            return preds[:, 0]

    def copy(self, *args, **kwargs):
        return super(ImgDiscx, self).copy(n_batch_logits=self._np_n_batch_logits)


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
        return super(ImgDiscz, self).copy(n_batch_logits=self._np_n_batch_logits)
