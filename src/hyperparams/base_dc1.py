import tensorflow as tf


class Hyperparams:  # change
    dtype = tf.float32

    input_size = 2

    input_height = 32
    input_width = 32
    input_channel = 3

    z_size = 100

    lr_autoencoder = 0.0001
    lr_decoder = 0.0001
    lr_disc = 0.0001
    beta1 = 0.9
    beta2 = 0.99

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']

    train_generator_adv = False

    model = 'bcgan'
    exp_name = 'bcgan_0'
