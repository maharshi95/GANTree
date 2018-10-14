from abc import ABCMeta, abstractmethod, abstractproperty
import tensorflow as tf


class BaseModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def step_train_autoencoder(self, inputs):
        return NotImplemented

    @abstractmethod
    def step_train_adv_generator(self, inputs):
        return NotImplemented

    @abstractmethod
    def step_train_discriminator(self, inputs):
        return NotImplemented

    @abstractmethod
    def compute_losses(self, inputs, losses):
        return NotImplemented

    @abstractmethod
    def encode(self, x):
        return NotImplemented

    @abstractmethod
    def decode(self, z):
        return NotImplemented

    @abstractmethod
    def reconstruct_x(self, x):
        return NotImplemented

    @abstractmethod
    def reconstruct_z(self, z):
        return NotImplemented

    @abstractmethod
    def discriminate(self, z):
        return NotImplemented

    @abstractproperty
    def network_loss_variables(self):
        # type: () -> list[tf.Summary]
        return NotImplemented
