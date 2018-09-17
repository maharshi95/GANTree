import tensorflow as tf


class Hyperparams:
    """
    Base Hyperparams class.
    It uses base version of bcgan with 1D x space and z space
    """
    dtype = tf.float32

    input_size = 1

    z_size = 1

    logit_batch_size = 100

    lr_autoencoder = 0.0005
    lr_decoder = 0.0005
    lr_disc = 0.0005

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']

    show_visual_while_training = True

    train_generator_adv = True
    train_autoencoder = True

    train_batch_logits = True
    train_sample_logits = True

    start_tensorboard = True
    base_port = 7001

    gen_iter_count = 10
    disc_iter_count = 30
    gan_switching_criteria = 'dynamic'  # ['fixed' / 'dynamic']

    model = 'bcgan'
    exp_name = 'bcgan_0'
    dataloader = 'two_gaussian'
