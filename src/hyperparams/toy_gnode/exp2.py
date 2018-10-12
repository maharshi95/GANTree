import numpy as np
import tensorflow as tf
from hyperparams import base


class Hyperparams(base.Hyperparams):
    dtype = tf.float32

    input_size = 2

    z_size = 3

    batch_size = 128
    logit_batch_size = 64

    lr_autoencoder = 0.0001
    lr_decoder = 0.0001
    lr_disc = 0.0001

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']

    z_bounds = 8.0

    z_means = 3 * np.ones(z_size)
    z_cov = np.eye(z_size)

    show_visual_while_training = True

    train_generator_adv = True
    train_autoencoder = True

    train_batch_logits = True
    train_sample_logits = True

    dataloader = 'four_gaussian_sym'

    n_modes = 4

    model = 'bcgnode'
    exp_name = '4_gaussians_bcgan_3D'
