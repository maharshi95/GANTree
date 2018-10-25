from __future__ import print_function

from collections import namedtuple

import numpy as np
import torch as tr
from torch import nn
from torch import optim

from configs import Config
from models import losses
from exp_context import ExperimentContext
from modules.activations import CircularTanH
from utils.decorators import make_tensor

H = ExperimentContext.Hyperparams

from modules.commons import NLinear, ZTransform
from base.model import BaseModel, BaseGan

n_batch_logits = H.logit_batch_size


class ToyEncoder(BaseModel):
    def __init__(self, in_feat=2, out_feat=2, out_scale=1.0):
        super(ToyEncoder, self).__init__()
        n_units = [32, 64, 128, 64, 32, out_feat]
        self.linear = NLinear(in_feat, n_units, act=nn.ELU)
        # self.act = CircularTanH()
        self.act = nn.Tanh()
        self.out_scale = out_scale
        self.init_params()

    def forward(self, x):
        x = self.linear(x)
        return self.out_scale * self.act(x / self.out_scale)


class ToyDecoder(BaseModel):
    def __init__(self, in_feat=2, out_feat=2):
        super(ToyDecoder, self).__init__()
        n_units = [32, 64, 128, 64, 32, out_feat]
        self.linear = NLinear(in_feat, n_units, act=nn.ELU)
        self.init_params()

    def forward(self, x):
        x = self.linear(x)
        return x


class ToyDisc(BaseModel):
    def __init__(self, in_feat=2):
        super(ToyDisc, self).__init__()

        n_common_features = 32

        self.common_block = NLinear(in_feat, [64, 64, 128, 128, n_common_features], act=nn.ELU)
        self.sample_logit_block = NLinear(n_common_features, [32, 1])
        self.batch_logit_block = NLinear(n_common_features * n_batch_logits, [64, 32, 1])

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
    def __init__(self, name, z_op_params, z_ip_params=None, encoder=None, decoder=None, disc=None, z_bounds=1.0):
        super(ToyGAN, self).__init__(name)

        if isinstance(z_op_params, int):
            z_op_params = tr.zeros(z_op_params), tr.eye(z_op_params)

        self.op_means, self.op_cov = z_op_params
        self.z_size = self.op_means.shape[0]

        self.transform = ZTransform(z_op_params, z_ip_params)

        self.encoder = encoder or ToyEncoder(out_feat=self.z_size, out_scale=z_bounds)
        self.decoder = decoder or ToyDecoder()
        self.disc = disc or ToyDisc()

        if Config.use_gpu:
            self.cuda()

        self.opt = {
            'encoder': optim.Adam(self.encoder.parameters()),
            'decoder': optim.Adam(self.decoder.parameters()),
            'disc': optim.Adam(self.disc.parameters()),
        }

    @property
    def z_op_params(self):
        return self.transform.src_params

    @property
    def z_ip_params(self):
        return self.transform.target_params

    def forward(self, *args):
        return super(ToyGAN, self).forward(*args)

    def sample(self, sample_shape, dist='in', z_bounds=4.0):
        params = self.z_ip_params if dist == 'in' else self.z_op_params
        f = tr.distributions.MultivariateNormal(*params)
        return f.sample(sample_shape)

    def classify(self, x):
        sample_logits, _ = self.disc(x)
        return 1 * (sample_logits >= 0.)

    def get_accuracies(self, x, z):
        with tr.no_grad():
            real_labels = self.classify(x)
            fake_labels = self.classify(self.decoder(z))

            gen_accuracy = 100 * (fake_labels == 1).type(tr.float32).mean()
            disc_accuracy = 50 * ((fake_labels == 0).type(tr.float32).mean() + (real_labels == 1).type(tr.float32).mean())
            return gen_accuracy, disc_accuracy

    ### Losses
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

        loss = sample_x_entropy_loss  # + batch_x_entropy_loss

        return loss

    def gen_adv_loss(self, z):
        sample_logits_fake, batch_logits_fake = self.disc(self.decoder(z))

        sample_loss = losses.sigmoid_cross_entropy_loss(sample_logits_fake, 1.0)
        batch_loss = losses.sigmoid_cross_entropy_loss(batch_logits_fake, 1.0)

        loss = sample_loss + batch_loss

        wgan_loss = tr.mean(sample_logits_fake)

        return loss

    def x_recon_loss(self, x):
        x_recon = self.decoder(self.transform(self.encoder(x)))
        x_recon_loss = tr.mean((x - x_recon) ** 2)
        return x_recon_loss

    def z_recon_loss(self, z):
        z_recon = self.transform(self.encoder(self.decoder(z)))
        z_recon_loss = tr.mean((z - z_recon) ** 2)
        return z_recon_loss

    def cyclic_loss(self, x, z):
        c_loss = self.x_recon_loss(x) + self.z_recon_loss(z)
        return c_loss

    #### Train Methods
    def step_train_discriminator(self, x, z):
        self.opt['disc'].zero_grad()
        loss = self.disc_adv_loss(x, z)
        loss.backward()

        self.opt['disc'].step()
        return loss

    def step_train_autoencoder(self, x, z):
        self.opt['encoder'].zero_grad()
        self.opt['decoder'].zero_grad()

        # loss = self.cyclic_loss(x, z)
        loss = self.cyclic_loss(x, z)
        loss.backward()

        self.opt['encoder'].step()
        self.opt['decoder'].step()

        return loss

    def step_train_encoder(self, x, z):
        self.opt['encoder'].zero_grad()

        loss = self.cyclic_loss(x, z)
        loss.backward()

        self.opt['encoder'].step()
        return loss

    def step_train_decoder(self, x, z):
        self.opt['decoder'].zero_grad()

        loss = self.cyclic_loss(x, z)
        loss.backward()

        self.opt['decoder'].step()
        return loss

    def step_train_generator(self, x, z):
        self.opt['decoder'].zero_grad()
        loss = self.gen_adv_loss(z)  # + self.cyclic_loss(x, z)
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
            g_adv_loss, d_adv_loss = self.gen_adv_loss(z), self.disc_adv_loss(x, z)

            return {
                'loss_x_recon': x_recon_loss,
                'loss_z_recon': z_recon_loss,
                'loss_cyclic': c_loss,
                'loss_gen': g_adv_loss,
                'loss_disc': d_adv_loss,
                'accuracy_gen': g_acc,
                'accuracy_disc': d_acc,
            }

    # DO NOT Use below functions for writing training procedures
    @make_tensor(use_gpu=Config.use_gpu)
    def encode(self, x_batch, transform=True):
        with tr.no_grad():
            z = self.encoder(x_batch)
            return self.transform(z) if transform else z

    @make_tensor(use_gpu=Config.use_gpu)
    def decode(self, z_batch):
        with tr.no_grad():
            return self.decoder(z_batch)

    @make_tensor(use_gpu=Config.use_gpu)
    def reconstruct_x(self, x_batch):
        with tr.no_grad():
            return self.decoder(self.transform(self.encoder(x_batch)))

    @make_tensor(use_gpu=Config.use_gpu)
    def reconstruct_z(self, z_batch, transform=True):
        with tr.no_grad():
            z_ = self.encoder(self.decoder(z_batch))
            return self.transform(z_) if transform else z_

    @make_tensor(use_gpu=Config.use_gpu)
    def discriminate(self, x_batch, split=True):
        with tr.no_grad():
            preds = self.disc.discriminate(x_batch)  # sample_logits_real
            if split:
                x_batch_real = x_batch[np.where(preds == 1)]
                x_batch_fake = x_batch[np.where(preds == 0)]
                return preds, x_batch_real, x_batch_fake
            else:
                return preds
