import tensorflow as tf


class Hyperparams:
    dtype = tf.float32

    input_size = 2

    z_size = 1

    lr_autoencoder = 0.0001
    lr_decoder = 0.0001
    lr_disc = 0.0001
