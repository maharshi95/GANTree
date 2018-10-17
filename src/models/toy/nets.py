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

    def discriminate(self, x):
        with tr.no_grad():
            inter = self.common_block(x)
            sample_logits = self.sample_logit_block(inter)
            preds = sample_logits >= 0.
            return preds[:, 0]


class ToyGAN(BaseGan):
    def __init__(self, name, encoder=None, decoder=None, disc=None):
        super(ToyGAN, self).__init__(name)
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
        return 1 * (sample_logits >= 0.)

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

    def x_recon_loss(self, x):
        x_recon = self.decoder(self.encoder(x))
        x_recon_loss = tr.mean((x - x_recon) ** 2)
        return x_recon_loss

    def z_recon_loss(self, z):
        z_recon = self.encoder(self.decoder(z))
        z_recon_loss = tr.mean((z - z_recon) ** 2)
        return z_recon_loss

    def cyclic_loss(self, x, z):
        c_loss = self.x_recon_loss(x) + self.z_recon_loss(z)
        return c_loss

    def step_train_discriminator(self, x, z):
        self.opt['disc'].zero_grad()
        loss = self.disc_adv_loss(x, z)
        loss.backward()

        self.opt['disc'].step()
        return loss

    def step_train_autoencoder(self, x, z):
        self.opt['encoder'].zero_grad()
        self.opt['decoder'].zero_grad()

        c_loss = self.cyclic_loss(x, z)

        loss = c_loss
        loss.backward()

        self.opt['encoder'].step()
        self.opt['decoder'].step()

        return loss

    def step_train_generator(self, z):
        self.opt['decoder'].zero_grad()
        loss = self.gen_adv_loss(z)
        loss.backward()
        self.opt['decoder'].step()
        return loss

    def step_train_x_clf(self, x1, x2, mu1, mu2, cov1, cov2):
        self.opt['encoder'].zero_grad()
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        x_clf_loss = losses.x_clf_loss(mu1, cov1, mu2, cov2, z1, z2)

        x_clf_loss.backward()

        self.opt['encoder'].step()

        return x_clf_loss

    def compute_metrics(self, x, z):
        with tr.no_grad():
            x_recon_loss = self.x_recon_loss(x)
            z_recon_loss = self.z_recon_loss(z)
            c_loss = x_recon_loss + z_recon_loss
            g_acc, d_acc = self.get_accuracies(x, z)

            return {
                'loss_x_recon': x_recon_loss,
                'loss_z_recon': z_recon_loss,
                'loss_cyclic': c_loss,
                'accuracy_gen': g_acc,
                'accuracy_disc': d_acc,
            }

    # DO NOT Use below functions for writing training procedures
    def encode(self, x_batch):
        if len(x_batch)==0:
            return tr.tensor([])

        with tr.no_grad():
            return self.encoder(x_batch)

    def decode(self, z_batch):
        with tr.no_grad():
            return self.decoder(z_batch)

    def reconstruct_x(self, x_batch):
        with tr.no_grad():
            return self.decoder(self.encoder(x_batch))

    def reconstruct_z(self, z_batch):
        with tr.no_grad():
            return self.encoder(self.decoder(z_batch))

    def discriminate(self, x_batch, split=True):
        with tr.no_grad():
            preds = self.disc.discriminate(x_batch)  # sample_logits_real
            if split:
                x_batch_real = x_batch[np.where(preds == 1)]
                x_batch_fake = x_batch[np.where(preds == 0)]
                return preds, x_batch_real, x_batch_fake
            else:
                return preds
