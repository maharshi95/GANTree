from __future__ import print_function
from collections import OrderedDict
import traceback
import numpy as np
import torch as tr
from torch import nn
from torch import optim

from models import losses
from exp_context import ExperimentContext

H = ExperimentContext.Hyperparams

from modules.commons import NLinear
from base.model import BaseModel, BaseGan

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


class ToyGAN(BaseGan):
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

    def classify(self, x):
        sample_logits, _ = self.disc(x)
        return 1 * (sample_logits >= 0.5)

    def get_accuracies(self, x, z):
        with tr.no_grad():
            real_labels = self.classify(x)
            fake_labels = self.classify(self.decoder(z))

            gen_accuracy = 100 * (fake_labels == 1).type(tr.FloatTensor).mean()
            disc_accuracy = 50 * ((fake_labels == 0).type(tr.FloatTensor).mean() + (real_labels == 1).type(tr.FloatTensor).mean())
        return gen_accuracy, disc_accuracy

    def disc_adv_loss(self, x, z):
        sample_logits_real, batch_logits_real = self.disc(x)
        sample_logits_fake, batch_logits_fake = self.disc(self.decoder(z))

        sample_loss_real = losses.sigmoid_cross_entropy_loss(sample_logits_real, 1.0)
        sample_loss_fake = losses.sigmoid_cross_entropy_loss(sample_logits_fake, 0.0)

        batch_loss_real = losses.sigmoid_cross_entropy_loss(batch_logits_real, 1.0)
        batch_loss_fake = losses.sigmoid_cross_entropy_loss(batch_logits_fake, 0.0)

        sample_x_entropy_loss = sample_loss_real + sample_loss_fake
        batch_x_entropy_loss = batch_loss_real + batch_loss_fake

        wgan_loss = tr.mean(sample_logits_real - sample_logits_fake)

        loss = sample_x_entropy_loss + batch_x_entropy_loss

        return loss

    def gen_adv_loss(self, z):
        sample_logits_fake, batch_logits_fake = self.disc(self.decoder(z))

        sample_loss = losses.sigmoid_cross_entropy_loss(sample_logits_fake, 1.0)
        batch_loss = losses.sigmoid_cross_entropy_loss(batch_logits_fake, 1.0)

        loss = sample_loss + batch_loss

        wgan_loss = tr.mean(sample_logits_fake)

        return loss

    def cyclic_loss(self, x, z):
        x_recon = self.decoder(self.encoder(x))
        z_recon = self.encoder(self.decoder(z))

        x_recon_loss = tr.mean((x - x_recon) ** 2)
        z_recon_loss = tr.mean((z - z_recon) ** 2)

        c_loss = x_recon_loss + z_recon_loss

        return c_loss

    def step_train_discriminator(self, x, z):
        loss = self.disc_adv_loss(x, z)

        self.opt['disc'].zero_grad()
        loss.backward()

        self.opt['disc'].step()
        return loss

    def step_train_autoencoder(self, x, z):
        c_loss = self.cyclic_loss(x, z)

        loss = c_loss

        self.opt['encoder'].zero_grad()
        self.opt['decoder'].zero_grad()

        loss.backward()

        self.opt['encoder'].step()
        self.opt['decoder'].step()

        return loss

    def step_train_generator(self, z):
        loss = self.gen_adv_loss(z)
        self.opt['decoder'].zero_grad()
        loss.backward()
        self.opt['decoder'].step()
        return loss

    def step_train_x_clf(self, x1, x2, mu1, mu2, cov1, cov2):
        x_clf_loss = losses.x_clf_loss(mu1, cov1, mu2, cov2, x1, x2)

        self.opt['encoder'].zero_grad()

        x_clf_loss.backward()

        self.opt['encoder'].step()

        return x_clf_loss
