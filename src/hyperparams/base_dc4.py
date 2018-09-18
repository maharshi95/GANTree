import tensorflow as tf
from . import base


class Hyperparams(base.Hyperparams):  # change
    dtype = tf.float32
    batch_size_train = 128
    batch_size_test = 128
    logit_batch_size = 32  # factor of batch_size

    input_size = 2

    input_height = 32
    input_width = 32
    input_channel = 1

    z_size = 100

    lr_autoencoder = 0.0001
    lr_decoder = 0.0001
    lr_disc = 0.0001
    beta1 = 0.6
    beta2 = 0.99

    z_dist_type = 'uniform'  # ['uniform', 'normal', 'sphere']
    show_visual_while_training = False

    z_bound = 1

    train_generator_adv = False
    train_autoencoder = True

    train_batch_logits = True
    train_sample_logits = True
    gen_iter_count = 20
    disc_iter_count = 30
    combined_iter_count = gen_iter_count + disc_iter_count

    model = 'dcgan'
    exp_name = 'autoencoder'
    dataloader = 'mnist'
