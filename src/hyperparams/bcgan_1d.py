import tensorflow as tf
from . import base


class Hyperparams(base.Hyperparams):
    dtype = tf.float32

    input_size = 1

    z_size = 1

    lr_autoencoder = 0.0001
    lr_decoder = 0.0001
    lr_disc = 0.0001

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']

    train_generator_adv = True

    model = 'bcgan_v2'
    exp_name = 'bcgan_0'
