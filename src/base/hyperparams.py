import torch as tr


class Hyperparams:
    """
    Base Hyperparams class.
    It uses base version of bcgan with 1D x space and z space
    """
    dtype = tr.float32

    # Trainer parameters:
    n_iterations = 2000

    show_visual_while_training = True
    train_generator_adv = True
    train_autoencoder = True

    train_batch_logits = True
    train_sample_logits = True

    start_tensorboard = True
    base_port = 7001

    gen_iter_count = 10
    disc_iter_count = 30
    step_ratio = gen_iter_count, disc_iter_count

    gan_switching_criteria = 'dynamic'  # ['fixed' / 'dynamic']

    # Dimension Parameters
    batch_size = 128
    logit_batch_size = 64

    input_size = 2
    z_size = 2
    z_bounds = 4.0

    # Learning Parameters
    lr_autoencoder = 0.0005
    lr_decoder = 0.0005
    lr_disc = 0.0005

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']

    model = 'bcgan'
    exp_name = 'trial_with_gmms'
    dataloader = 'four_gaussian_sym'


    n_child_nodes = 2

    child_iter = 50




