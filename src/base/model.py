from __future__ import print_function

import os

import torch as tr
from torch import nn
from torch.nn import init

from paths import Paths
from utils import bash_utils


class BaseModel(nn.Module):
    def __init__(self, name=None):
        super(BaseModel, self).__init__()
        self.name = str(self) if name is None else name

    def copy(self, *args, **kwargs):
        module = self.__class__(*args, **kwargs)
        module.load_state_dict(self.state_dict())
        return module

    def init_params(self):
        def init_fn(module):
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
                init.constant_(module.bias, 0.001)

            elif isinstance(module, nn.Conv2d):
                init.xavier_normal_(module.weight)
                init.constant_(module.bias, 0.001)

            elif isinstance(module, nn.ConvTranspose2d):
                init.xavier_normal_(module.weight)
                init.constant_(module.bias, 0.001)

            elif isinstance(module, nn.BatchNorm2d):
                init.normal_(module.weight, 1.0, 0.02)
                init.constant_(module.bias, 0.0)

        self.apply(init_fn)

    def get_params_path(self, dir_name, weight_label, iter_no=None):
        return Paths.get_saved_params_path(dir_name, self.name, weight_label, iter_no) + '.pt'

    def get_params_dir_path(self, weight_label):
        return Paths.get_params_dir_path(weight_label, self.name)

    def get_log_writer_path(self, writer_name):
        return Paths.log_writer_path(writer_name, self.name)

    def create_params_dir(self):
        bash_utils.create_dir(self.get_params_dir_path('iter'))

    def save_params(self, dir_name, weight_label, iter_no=None):
        state_dict = {'iteration': iter_no,
                      'state_dict': {'encoder': self.encoder.state_dict(),
                                     'generator': self.generator.state_dict(),
                                     'discriminator': self.discriminator.state_dict()}
                      # 'optimizer' : {'optimizer_G': self.optimizer_G.state_dict(),
                      #                'optimizer_D': self.optimizer_D.state_dict()}
                    }        
        path = self.get_params_path(dir_name, weight_label, iter_no)
        tr.save(state_dict, path)

    def load_params(self, dir_name, weight_label, iter_no=None, path = None):
        if not path:
            path = self.get_params_path(dir_name, weight_label, iter_no)
        checkpoint = tr.load(path)
    
        iter_no = checkpoint['iteration']
        
        self.encoder.load_state_dict(checkpoint['state_dict']['encoder'])
        self.generator.load_state_dict(checkpoint['state_dict']['generator'])
        self.discriminator.load_state_dict(checkpoint['state_dict']['discriminator'])
        
        # self.optimizer_D.load_state_dict(checkpoint['optimizer']['optimizer_D'])
        # self.optimizer_G.load_state_dict(checkpoint['optimizer']['optimizer_G'])

class BaseGan(BaseModel):
    def __init__(self, name=None):
        super(BaseGan, self).__init__(name)

    def step_train_generator(self, x, z):
        return NotImplementedError

    def step_train_discriminator(self, x, z):
        return NotImplementedError

    def step_train_autoencoder(self, x, z):
        return NotImplementedError

    def step_train_encoder(self, x, z):
        return NotImplementedError

    def step_train_decoder(self, x, z):
        return NotImplementedError
