from __future__ import print_function
from collections import OrderedDict
import traceback
import numpy as np
import torch as tr
from torch import nn
from torch import optim

from exp_context import ExperimentContext

H = ExperimentContext.Hyperparams

from modules.commons import NLinear
from base.model import BaseModel

n_batch_logits = H.logit_batch_size


class ToyEncoder(NLinear, BaseModel):
    def __init__(self, in_feat=2, out_feat=2):
        n_units = [32, 64, 128, 64, 32, out_feat]
        super(ToyEncoder, self).__init__(in_feat, n_units, act=nn.ELU)
        self.init_params()


class ToyDecoder(NLinear, BaseModel):
    def __init__(self, in_feat=2, out_feat=2):
        n_units = [32, 64, 128, 64, 32, out_feat]
        super(ToyDecoder, self).__init__(in_feat, n_units, act=nn.ELU)
        self.init_params()


class ToyDisc(BaseModel):
    def __init__(self, in_feat=2):
        super(ToyDisc, self).__init__()

        common_n_features = 32

        self.common_block = NLinear(in_feat, [64, 64, 128, 128, common_n_features])
        self.sample_logit_block = NLinear(common_n_features, [32, 1])
        self.batch_logit_block = NLinear(common_n_features * n_batch_logits, [64, 32, 1])

        self.init_params()

    def forward(self, x):
        inter = self.common_block(x)
        sample_logits = self.sample_logit_block(inter)
        inter_view = inter.view(inter.shape[0] / n_batch_logits, -1)
        batch_logits = self.batch_logit_block(inter_view)
        return sample_logits, batch_logits


class ToyGAN(nn.Module):
    def __init__(self, name, encoder=None, decoder=None, disc=None):
        super(ToyGAN, self).__init__()
        self.name = name
        self.encoder = encoder or ToyEncoder()
        self.decoder = decoder or ToyDecoder()
        self.disc = disc or ToyDisc()

        self.opt = {
            'encoder': optim.Adam(self.encoder.parameters()),
            'decoder': optim.Adam(self.decoder.parameters()),
            'disc': optim.Adam(self.disc.parameters()),
        }

    def forward(self, *args):
        return super(ToyGAN, self).forward(*args)

    def step_train_disc(self, x, z):
        x_real = x
        x_fake = self.decoder(z)

        sample_logits_real, batch_logits_real = self.disc(x_real)
        sample_logits_fake, batch_logits_fake = self.disc(x_fake)

        loss = tr.mean(sample_logits_real - sample_logits_fake)

        self.opt['disc'].zero_grad()

        loss.backward()

        self.opt['disc'].step()
        return loss

    def step_train_generator(self, x, z):
        x_real = x
        z_rand = z

        z_real = self.encoder(x_real)
        x_recon = self.decoder(z_real)

        x_fake = self.decoder(z_rand)
        z_recon = self.encoder(x_fake)

        x_loss = tr.mean((x_real - x_recon) ** 2)
        z_loss = tr.mean((z_real - z_recon) ** 2)

        c_loss = x_loss + z_loss

        self.opt['encoder'].zero_grad()
        self.opt['decoder'].zero_grad()

        c_loss.backward()

        self.opt['encoder'].step()
        self.opt['decoder'].step()

        return c_loss

    def x_clf_loss(self, z1, z2, means1, means2, cov1, cov2):
        f1 = tr.distributions.MultivariateNormal(means1, cov1)
        f2 = tr.distributions.MultivariateNormal(means2, cov2)

        loss = (f1.log_prob(z1) - f2.log_prob(z1)).mean() + (f2.log_prob(z2) - f1.log_prob(z2)).mean()

        return loss

    def step_train_encoder(self, x1, x2, means1, means2, cov1, cov2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        clf_loss = self.x_clf_loss(z1, z2, means1, means2, cov1, cov2)
        self.opt['encoder'].zero_grad()
        clf_loss.backward()

        self.opt['encoder'].step()

        return clf_loss


