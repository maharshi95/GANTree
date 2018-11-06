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


class ImgEncoder(BaseModel):
    def __init__(self, out_scale=4.0):
        super(ImgEncoder, self).__init__()

        self._np_out_scale = out_scale
        self.out_scale = Parameter(tr.tensor(out_scale), requires_grad=False)

        def cnn_block(in_filters, out_filters, bn=True, kernel_size=3, stride=2):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))

            return block

        self.conv1 = nn.Sequential(*cnn_block(H.input_channel, 32, bn=False, kernel_size=5, stride=1))
        self.conv2 = nn.Sequential(*cnn_block(32, 64, kernel_size=5, stride=2))
        self.conv3 = nn.Sequential(*cnn_block(64, 128, kernel_size=5, stride=1))
        self.conv4 = nn.Sequential(*cnn_block(128, 256, kernel_size=5, stride=2))
        self.conv5 = nn.Sequential(*cnn_block(256, 512, kernel_size=5, stride=1))
        self.conv6 = nn.Sequential(*cnn_block(512, 512, kernel_size=5, stride=2))

        # The height and width of downsampled image
        linear_size = H.input_height  # // 2 ** 3
        self.layer = nn.Linear(512 * (linear_size ** 2), H.z_size)

        self.init_params()

    @property
    def z_bounds(self):
        return self._np_out_scale

    def forward(self, img):
        img = F.pad(img, (2, 2, 2, 2))
        conv1 = self.conv1(img)
        print(conv1.shape)

        conv1 = F.pad(conv1, (18, 18, 18, 18))
        conv2 = self.conv2(conv1)
        print(conv2.shape)

        conv2 = F.pad(conv2, (2, 2, 2, 2))
        conv3 = self.conv3(conv2)
        print(conv3.shape)

        conv3 = F.pad(conv3, (18, 18, 18, 18))
        conv4 = self.conv4(conv3)
        print(conv4.shape)

        conv4 = F.pad(conv4, (2, 2, 2, 2))
        conv5 = self.conv5(conv4)
        print(conv5.shape)

        conv5 = F.pad(conv5, (18, 18, 18, 18))
        conv6 = self.conv6(conv5)
        print(conv6.shape)

        out = conv6.view(conv6.size(0), -1)
        z = self.layer(out)
        z = self.out_scale * F.tanh(z / self.out_scale)
        print(z.shape)
        return z

    def copy(self, *args, **kwargs):
        return super(ImgEncoder, self).copy(out_scale=self._np_out_scale)


class ImgDecoder(BaseModel):
    def __init__(self, out_scale=4.0):
        super(ImgDecoder, self).__init__()

        self._np_out_scale = out_scale
        self.out_scale = Parameter(tr.tensor(out_scale), requires_grad=False)

        # ToDo add upsample block
        def transpose_conv2d(in_channels, out_channels, bn=True, kernel_size=5, stride=2, output_padding=(1, 1), padding=(2, 2)):
            block = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                        output_padding=output_padding)]
            if bn:
                block.append(nn.BatchNorm2d(out_channels, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))

            return block

        self.linear = nn.Linear(H.z_size, 4 * 4 * 256)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.tconv1 = nn.Sequential(*transpose_conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2))
        self.tconv2 = nn.Sequential(*transpose_conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=2))
        self.tconv3 = nn.Sequential(*transpose_conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2))
        self.tconv4 = nn.Sequential(
            *transpose_conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, output_padding=(0, 0)))
        self.tconv5 = nn.Sequential(
            *transpose_conv2d(in_channels=32, out_channels=H.input_channel, kernel_size=5, stride=1, output_padding=(0, 0)))

        self.init_params()

    @property
    def z_bounds(self):
        return self._np_out_scale

    def forward(self, z):
        fc1 = self.linear(z)
        fc1 = fc1.view(z.shape[0], 256, 4, 4)
        fc1 = self.lrelu(fc1)
        print(fc1.shape)

        tconv1 = self.tconv1(fc1)
        print(tconv1.shape)

        tconv2 = self.tconv2(tconv1)
        print(tconv2.shape)

        tconv3 = self.tconv3(tconv2)
        print(tconv3.shape)

        tconv4 = self.tconv4(tconv3)
        print(tconv4.shape)

        tconv5 = self.tconv5(tconv4)
        print(tconv5.shape)

        return tconv5
        # return  self.out_scale * F.tanh(tconv5 / self.out_scale)

    def copy(self, *args, **kwargs):
        return super(ImgEncoder, self).copy(out_scale=self._np_out_scale)


class ImgDiscx(BaseModel):

    def __init__(self, n_batch_logits):
        super(ImgDiscx, self).__init__()
        self._np_n_batch_logits = n_batch_logits
        self.n_batch_logits = Parameter(tr.tensor(n_batch_logits), requires_grad=False)

        def cnn_block(in_filters, out_filters, bn=True, kernel_size=3, stride=2):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))

            return block

        self.conv1 = nn.Sequential(*cnn_block(H.input_channel, 32, bn=False, kernel_size=5, stride=1))
        self.conv2 = nn.Sequential(*cnn_block(32, 64, kernel_size=5, stride=2))
        self.conv3 = nn.Sequential(*cnn_block(64, 128, kernel_size=5, stride=1))
        self.conv4 = nn.Sequential(*cnn_block(128, 256, kernel_size=5, stride=2))
        self.conv5 = nn.Sequential(*cnn_block(256, 512, kernel_size=5, stride=1))
        self.conv6 = nn.Sequential(*cnn_block(512, 512, kernel_size=5, stride=2))

        # The height and width of downsampled image
        linear_size = H.input_height  # // 2 ** 3
        self.layer = nn.Linear(512 * (linear_size ** 2), H.z_size)

        self.sample_logit_block = NLinear(H.z_size, [64, 32, 1])
        self.batch_logit_block = NLinear(H.z_size * n_batch_logits, [16, 16, 1])

        self.init_params()

    def forward(self, img):
        img = F.pad(img, (2, 2, 2, 2))
        conv1 = self.conv1(img)
        print(conv1.shape)

        conv1 = F.pad(conv1, (18, 18, 18, 18))
        conv2 = self.conv2(conv1)
        print(conv2.shape)

        conv2 = F.pad(conv2, (2, 2, 2, 2))
        conv3 = self.conv3(conv2)
        print(conv3.shape)

        conv3 = F.pad(conv3, (18, 18, 18, 18))
        conv4 = self.conv4(conv3)
        print(conv4.shape)

        conv4 = F.pad(conv4, (2, 2, 2, 2))
        conv5 = self.conv5(conv4)
        print(conv5.shape)

        conv5 = F.pad(conv5, (18, 18, 18, 18))
        conv6 = self.conv6(conv5)
        print(conv6.shape)

        out = conv6.view(conv6.size(0), -1)
        z_out = self.layer(out)

        print(z_out.shape)
        z_out_view = z_out.view(-1, H.z_size * self.n_batch_logits)

        sample_logits = self.sample_logit_block(z_out)
        batch_logits = self.batch_logit_block(z_out_view)

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
