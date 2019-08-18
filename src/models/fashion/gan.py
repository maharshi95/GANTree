import numpy as np
import torch as tr
from torch import optim
import logging
from base.hyperparams import Hyperparams
from base.model import BaseModel
from configs import Config
from exp_context import ExperimentContext
from models import losses
from .nets import Generator, Encoder, Discriminator
from modules.commons import ZTransform
from torch.autograd import Variable

from utils.decorators import make_tensor, tensorify

import itertools

H = ExperimentContext.Hyperparams  # type: Hyperparams
logger = logging.getLogger(__name__)


class ImgGAN(BaseModel):
    def __init__(self, name, z_op_params, z_ip_params = None, encoder = None, generator = None, discriminator = None):
        
        super(ImgGAN, self).__init__(name)

        logger.info('aae constructor entered')

        if isinstance(z_op_params, int):
            z_op_params = tr.zeros(z_op_params), tr.eye(z_op_params)

        self.op_means, self.op_cov = tr.Tensor(z_op_params[0]), tr.Tensor(z_op_params[1])

        self.z_dim = self.op_means.shape[0]

        self.transform = ZTransform(z_op_params, z_ip_params)

        self.epsilon = H.epsilon
        self.channel = H.channel

        self.encoder = encoder or Encoder(z_dim = H.z_dim, channel = self.channel)
        self.generator = generator or Generator(z_dim = H.z_dim, channel = self.channel)
        self.discriminator = discriminator or Discriminator(z_dim = H.z_dim)

        if Config.use_gpu:
            self.cuda()

        self.optimizer_G = optim.Adam(itertools.chain(self.encoder.parameters(), self.generator.parameters()), lr=H.lr, betas=(H.b1, H.b2))
        self.optimizer_D = optim.SGD(self.discriminator.parameters(), lr=H.lr)

        self.adversarial_loss = tr.nn.BCELoss()
        self.pixelwise_loss = tr.nn.L1Loss()

        logger.info('aae constructor created')

    @staticmethod
    def create_from_hyperparams(name, hyperparams, cov_sign):
        # type: (str, Hyperparams, str) -> ImgGAN
        z_op_params = hyperparams.z_means(), hyperparams.z_cov(sign = cov_sign)
        return ImgGAN(name, z_op_params)

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
  
    def g(self, z):
        return self.generator(z)
    
    def g_recon(self, x):
        return self.generator(self.encoder(x))
    
    def get_disc_x_accuracies(self, x, z, separate_acc=False):
        with tr.no_grad():
            real_labels_x = self.classify_z(z)
            fake_labels_x = self.classify_z(self.encoder(x))

            gen_x_accuracy = 100 * (fake_labels_x == 1).type(tr.float32).mean()
            disc_x_accuracy = 50 * ((fake_labels_x == 0).type(tr.float32).mean() + (real_labels_x == 1).type(tr.float32).mean())

            if (separate_acc):
                disc_x_real_acc = 100 * (real_labels_x == 1).type(tr.float32).mean()
                disc_x_fake_acc = 100 * (fake_labels_x == 0).type(tr.float32).mean()
                return gen_x_accuracy, disc_x_accuracy, disc_x_fake_acc, disc_x_real_acc

            return gen_x_accuracy, disc_x_accuracy
        
    def classify_z(self, z):
        sample_logits = self.discriminator(z)
        return 1 * (sample_logits >= 0.5)

    @make_tensor(use_gpu=Config.use_gpu)   
    def reconstruct_x(self, x_batch):
        with tr.no_grad():
            return self.generator(self.encoder(x_batch))
    
    @make_tensor(use_gpu=Config.use_gpu)
    def decode(self, z_batch):
        with tr.no_grad():
            return self.generator(z_batch)

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

    def sample(self, sample_shape, dist='in'):
        params = self.z_op_params if dist == 'in' else self.z_ip_params
        f = tr.distributions.MultivariateNormal(*params)
        return f.sample(sample_shape).cuda()


    def compute_metrics(self, x, z, disc_real_acc = False):
        valid = Variable(tr.Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(tr.Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False).cuda()
            
        with tr.no_grad():
            encoded_imgs = self.encoder(x)
            decoded_imgs = self.generator(encoded_imgs)
            
            # Loss measures generator's ability to fool the discriminator
            adv_loss = 0.001 * self.adversarial_loss(self.discriminator(encoded_imgs), valid)
            pix_loss = 0.999 * self.pixelwise_loss(decoded_imgs, x)
            g_loss = adv_loss + pix_loss

            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.adversarial_loss(self.discriminator(z), valid)
            fake_loss = self.adversarial_loss(self.discriminator(encoded_imgs), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            
            if (disc_real_acc):
                g_x_acc, d_x_acc, d_x_acc_fake, d_x_acc_real = self.get_disc_x_accuracies(x, z, True)
            else:
                g_x_acc, d_x_acc = self.get_disc_x_accuracies(x, z)

            dict_metrics = {
                'adv_loss': adv_loss,
                'pix_loss': pix_loss,
                'g_loss': g_loss,

                'real_loss': real_loss,
                'fake_loss': fake_loss,
                'd_loss': d_loss,
                
                'accuracy_gen_x': g_x_acc,
                'accuracy_dis_x': d_x_acc
            }

            return dict_metrics
