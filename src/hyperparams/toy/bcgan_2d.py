import tensorflow as tf
from base import hyperparams as base_hp


class Hyperparams(base_hp.Hyperparams):
    dtype = tf.float32

    batch_size = 64
    logit_batch_size = 32

    input_size = 2

    z_size = 2

    lr_autoencoder = 0.0001
    lr_decoder = 0.0001
    lr_disc = 0.0001

    z_dist_type = 'uniform'  # ['uniform', 'normal', 'sphere']

    z_bounds = 4.0

    # show_visual_while_training = True

    train_generator_adv = True
    train_autoencoder = True

    train_batch_logits = True
    train_sample_logits = True

    dataloader = 'four_gaussian_sym'

    model = 'bcgan'
    exp_name = 'trial'
