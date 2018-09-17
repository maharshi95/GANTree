import tensorflow as tf
from . import base


class Hyperparams(base.Hyperparams):
    dtype = tf.float32

    n_modes = 5

    input_size = 2

    z_size = 2

    lr_autoencoder = 0.0001
    lr_decoder = 0.0001
    lr_disc = 0.0001

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']

    show_visual_while_training = True

    train_generator_adv = True
    train_autoencoder = True

    model = 'bcgan_v2'
    exp_name = 'gantree-first-run'
