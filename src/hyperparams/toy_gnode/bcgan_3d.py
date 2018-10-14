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

    z_dist_type = 'sphere'  # ['uniform', 'normal', 'sphere']

    z_bounds = 1.0

    show_visual_while_training = True

    train_generator_adv = True
    train_autoencoder = True

    train_batch_logits = True
    train_sample_logits = True

    dataloader = 'two_gaussian'

    n_modes = 10

    model = 'bcgnode'
    exp_name = '4_gaussians_bcgan_3D'
