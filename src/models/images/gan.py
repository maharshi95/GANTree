import numpy as np
import torch as tr
from torch import optim

from base.hyperparams import Hyperparams
from base.model import BaseGan
from configs import Config
from exp_context import ExperimentContext
from models import losses
# from models.toy.nets import ToyEncoder, ToyDecoder, ToyDisc
from models.images.image_nets import ImgDiscx, ImgDiscz, ImgDecoder, ImgEncoder
from modules.commons import ZTransform

from utils.decorators import make_tensor, tensorify

H = ExperimentContext.Hyperparams  # type: Hyperparams


class ImgGAN(BaseGan):

    def __init__(self, name, z_op_params, z_ip_params=None,
                 encoder=None, decoder=None, disc_x=None, disc_z=None, z_bounds=H.z_bounds):
        super(ImgGAN, self).__init__(name)

        if isinstance(z_op_params, int):
            z_op_params = tr.zeros(z_op_params), tr.eye(z_op_params)

        self.op_means, self.op_cov = tr.Tensor(z_op_params[0]), tr.Tensor(z_op_params[1])

        self.z_size = self.op_means.shape[0]

        self.transform = ZTransform(z_op_params, z_ip_params)

        self.encoder = encoder or ImgEncoder(out_scale=z_bounds)
        self.decoder = decoder or ImgDecoder(out_scale=z_bounds)
        self.disc_x = disc_x or ImgDiscx(n_batch_logits=H.logit_x_batch_size)
        self.disc_z = disc_z or ImgDiscz(n_batch_logits=H.logit_z_batch_size)

        if Config.use_gpu:
            self.cuda()

        ed_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.opt = {
            'autoencoder': optim.Adam(ed_params),
            'generator': optim.Adam(ed_params),

            'encoder': optim.Adam(self.encoder.parameters()),
            'decoder': optim.Adam(self.decoder.parameters()),

            'encoder_c': optim.Adam(self.encoder.parameters()),
            'decoder_c': optim.Adam(self.decoder.parameters()),

            'disc_x': optim.Adam(self.disc_x.parameters()),
            'disc_z': optim.Adam(self.disc_z.parameters()),

            'case1': optim.Adam(self.encoder.parameters()),
            'case2': optim.Adam(self.decoder.parameters()),

        }

    @staticmethod
    def create_from_hyperparams(name, hyperparams, cov_sign='+'):
        # type: (str, Hyperparams, str) -> ImgGAN
        z_op_params = hyperparams.z_means(), hyperparams.z_cov(cov_sign)
        return ImgGAN(name, z_op_params, z_bounds=hyperparams.z_bounds)

    @property
    def z_op_params(self):
        return self.transform.src_params

    @property
    def z_ip_params(self):
        return self.transform.target_params

    @z_op_params.setter
    def z_op_params(self, value):
        self.transform.src_params = value

    @z_ip_params.setter
    def z_ip_params(self, value):
        self.transform.target_params = value

    @property
    def z_bounds(self):
        return self.encoder.z_bounds

    def forward(self, *args):
        return super(ImgGAN, self).forward(*args)

    def sample(self, sample_shape, dist='in', z_bounds=4.0):
        params = self.z_ip_params if dist == 'in' else self.z_op_params
        f = tr.distributions.MultivariateNormal(*params)
        return f.sample(sample_shape)

    def classify_x(self, x):
        sample_logits, _ = self.disc_x(x)
        return 1 * (sample_logits >= 0.)

    def classify_z(self, z):
        sample_logits, _ = self.disc_z(z)
        return 1 * (sample_logits >= 0.)

    def get_disc_z_accuracies(self, x, z):
        with tr.no_grad():
            real_labels_z = self.classify_z(z)
            fake_labels_z = self.classify_z(self.encoder(x))

            gen_z_accuracy = 100 * (fake_labels_z == 1).type(tr.float32).mean()
            disc_z_accuracy = 50 * ((fake_labels_z == 0).type(tr.float32).mean() + (real_labels_z == 1).type(tr.float32).mean())

            return gen_z_accuracy, disc_z_accuracy

    def get_disc_x_accuracies(self, x, z):
        with tr.no_grad():
            real_labels_x = self.classify_x(x)
            fake_labels_x = self.classify_x(self.decoder(z))

            gen_x_accuracy = 100 * (fake_labels_x == 1).type(tr.float32).mean()
            disc_x_accuracy = 50 * ((fake_labels_x == 0).type(tr.float32).mean() + (real_labels_x == 1).type(tr.float32).mean())

            return gen_x_accuracy, disc_x_accuracy

    def discriminative_x_entropy_loss(self, s_logits_real, s_logits_fake, b_logits_real, b_logits_fake):
        s_loss_real = losses.sigmoid_cross_entropy_loss(s_logits_real, 1.0)
        s_loss_fake = losses.sigmoid_cross_entropy_loss(s_logits_fake, 0.0)
        s_x_entropy_loss = (s_loss_real + s_loss_fake) / 2.0

        b_loss_real = losses.sigmoid_cross_entropy_loss(b_logits_real, 1.0)
        b_loss_fake = losses.sigmoid_cross_entropy_loss(b_logits_fake, 0.0)
        b_x_entropy_loss = (b_loss_real + b_loss_fake) / 2.0

        loss = s_x_entropy_loss  # + batch_x_entropy_loss
        if H.train_batch_logits:
            loss += b_x_entropy_loss

        return loss

    def generative_x_entropy_loss(self, sample_logits, batch_logits):
        sample_loss = losses.sigmoid_cross_entropy_loss(sample_logits, 1.0)
        batch_loss = losses.sigmoid_cross_entropy_loss(batch_logits, 1.0)

        loss = sample_loss  # + batch_loss
        if H.train_batch_logits:
            loss += batch_loss

        wgan_loss = tr.mean(sample_logits)

        return loss

    def disc_adv_loss_x(self, x, z):
        sample_logits_real, batch_logits_real = self.disc_x(x)
        sample_logits_fake, batch_logits_fake = self.disc_x(self.decoder(z))

        return self.discriminative_x_entropy_loss(sample_logits_real, sample_logits_fake, batch_logits_real, batch_logits_fake)

    def disc_adv_loss_z(self, x, z):
        sample_logits_real, batch_logits_real = self.disc_z(z)
        sample_logits_fake, batch_logits_fake = self.disc_z(self.encoder(x))
        return self.discriminative_x_entropy_loss(sample_logits_real, sample_logits_fake, batch_logits_real, batch_logits_fake)

    def gen_adv_loss_x(self, z):
        sample_logits_fake, batch_logits_fake = self.disc_x(self.decoder(z))
        return self.generative_x_entropy_loss(sample_logits_fake, batch_logits_fake)

    def gen_adv_loss_z(self, x):
        sample_logits_fake, batch_logits_fake = self.disc_z(self.encoder(x))
        return self.generative_x_entropy_loss(sample_logits_fake, batch_logits_fake)

    def x_recon_loss(self, x):
        # x_recon = self.decoder(self.transform(self.encoder(x)))
        x_recon = self.decoder(self.encoder(x))
        error_vectors = ((x - x_recon) ** 2).view(x.shape[0], -1)
        x_recon_loss = tr.mean(tr.sum(error_vectors, dim=-1))
        return x_recon_loss

    def z_recon_loss(self, z):
        # z_recon = self.transform(self.encoder(self.decoder(z)))
        z_recon = self.encoder(self.decoder(z))
        error_vectors = ((z - z_recon) ** 2).view(z.shape[0], -1)

        z_recon_loss = tr.mean(tr.sum(error_vectors, dim=-1))
        return z_recon_loss

    def cyclic_loss(self, x, z):
        c_loss = self.x_recon_loss(x) + self.z_recon_loss(z)
        return c_loss

    #### Train Methods

    def step_train_disc_x(self, x, z):
        self.opt['disc_x'].zero_grad()
        loss = self.disc_adv_loss_x(x, z)
        loss.backward()
        self.opt['disc_x'].step()
        return loss

    def step_train_disc_z(self, x, z):
        self.opt['disc_z'].zero_grad()
        loss = self.disc_adv_loss_z(x, z)
        loss.backward()
        self.opt['disc_z'].step()
        return loss

    def step_train_gen_x(self, z):
        self.opt['decoder'].zero_grad()
        loss_z = self.gen_adv_loss_x(z)
        loss_z.backward()
        self.opt['decoder'].step()
        return loss_z

    def step_train_gen_z(self, x):
        self.opt['encoder'].zero_grad()
        loss_x = self.gen_adv_loss_z(x)
        loss_x.backward()
        self.opt['encoder'].step()
        return loss_x

    def step_train_discriminator(self, x, z):
        loss_x = self.step_train_disc_x(x, z)
        # loss_z = self.step_train_disc_z(x, z)
        loss_z = loss_x
        return loss_x, loss_z

    def step_train_generator(self, x, z):
        loss_x = self.step_train_gen_x(z)
        # loss_z = self.step_train_gen_z(x)
        loss_z = loss_x
        return loss_x, loss_z

    def step_train_autoencoder(self, x, z):
        self.opt['encoder_c'].zero_grad()
        self.opt['decoder_c'].zero_grad()

        loss = self.cyclic_loss(x, z)
        # loss = self.x_recon_loss(x)
        loss.backward()

        self.opt['encoder_c'].step()
        self.opt['decoder_c'].step()

        return loss

    # Not Used
    def step_train_encoder(self, x, z, lam=0.0001):
        self.opt['encoder'].zero_grad()

        loss = self.cyclic_loss(x, z) + lam * self.gen_adv_loss_x(x)
        loss.backward()

        self.opt['encoder'].step()
        return loss

    # Not Used
    def step_train_decoder(self, x, z, lam=0.0001):
        self.opt['decoder'].zero_grad()

        loss = self.cyclic_loss(x, z) + lam * self.gen_adv_loss(z)
        loss.backward()

        self.opt['decoder'].step()
        return loss

    # Not Used
    def step_train_case1(self, x, lam=0.0001):
        self.opt['case1'].zero_grad()

        loss = self.x_recon_loss(x) + lam * self.gen_adv_loss_x(x)
        loss.backward()

        self.opt['case1'].step()
        return loss

    # Not Used
    def step_train_case2(self, z, lam=0.0001):
        self.opt['case2'].zero_grad()

        loss = self.z_recon_loss(z) + lam * self.gen_adv_loss(z)
        loss.backward()

        self.opt['case2'].step()
        return loss

    def compute_metrics(self, x, z):
        with tr.no_grad():
            x_recon_loss = self.x_recon_loss(x)
            z_recon_loss = self.z_recon_loss(z)
            c_loss = x_recon_loss + z_recon_loss

            g_x_acc, d_x_acc = self.get_disc_x_accuracies(x, z)
            g_z_acc, d_z_acc = self.get_disc_z_accuracies(x, z)

            g_adv_loss_x, d_adv_loss_x = self.gen_adv_loss_x(z), self.disc_adv_loss_x(x, z)
            g_adv_loss_z, d_adv_loss_z = self.gen_adv_loss_z(x), self.disc_adv_loss_z(x, z)

            return {
                'loss_x_recon': x_recon_loss,
                'loss_z_recon': z_recon_loss,
                'loss_cyclic': c_loss,

                'loss_adv_gen_x': g_adv_loss_x,
                'loss_adv_dis_x': d_adv_loss_x,

                'loss_adv_gen_z': g_adv_loss_z,
                'loss_adv_dis_z': d_adv_loss_z,

                'accuracy_gen_x': g_x_acc,
                'accuracy_dis_x': d_x_acc,

                'accuracy_gen_z': g_z_acc,
                'accuracy_dis_z': d_z_acc,
            }

    # DO NOT Use below functions for writing training procedures
    @make_tensor(use_gpu=Config.use_gpu)
    def encode(self, x_batch, transform=False, both=False):
        with tr.no_grad():
            z = self.encoder(x_batch)
            if not both and not transform:
                return z

            zt = self.transform(z)

            if not both and transform:
                return zt
            return z, zt

    @make_tensor(use_gpu=Config.use_gpu)
    def decode(self, z_batch):
        with tr.no_grad():
            return self.decoder(z_batch)

    @make_tensor(use_gpu=Config.use_gpu)
    def reconstruct_x(self, x_batch):
        with tr.no_grad():
            # return self.decoder(self.transform(self.encoder(x_batch)))
            return self.decoder(self.encoder(x_batch))

    @make_tensor(use_gpu=Config.use_gpu)
    def reconstruct_z(self, z_batch, transform=False):
        with tr.no_grad():
            z_ = self.encoder(self.decoder(z_batch))
            return self.transform(z_) if transform else z_

    @make_tensor(use_gpu=Config.use_gpu)
    def discriminate(self, x_batch, split=True):
        with tr.no_grad():
            z_batch = self.encoder(x_batch)
            preds = self.disc_x.discriminate(x_batch)  # sample_logits_real
            if split:
                x_batch_real = x_batch[np.where(preds == 1)]
                x_batch_fake = x_batch[np.where(preds == 0)]
                return preds, x_batch_real, x_batch_fake
            else:
                return preds
