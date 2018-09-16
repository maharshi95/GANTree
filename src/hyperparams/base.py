import tensorflow as tf


class Hyperparams:
    dtype = tf.float32

    input_size = 1

    z_size = 1

    lr_autoencoder = 0.0001
    lr_decoder = 0.0001
    lr_disc = 0.0001

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']
    
    show_visual_while_training = True

    train_generator_adv = True

    model = 'bcgan'
    exp_name = 'bcgan_0'
    dataloader = 'two_gaussian'
