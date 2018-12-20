from base import hyperparams


class Hyperparams(hyperparams.Hyperparams):  # change

    dtype = float

    # Trainer parameters:
    n_iterations = 100000

    show_visual_while_training = True
    train_generator_adv = True
    train_autoencoder = True

    train_batch_logits = True
    train_sample_logits = True

    start_tensorboard = True

    circular_bounds = False

    gen_iter_count = 20
    disc_iter_count = 40
    step_ratio = gen_iter_count, disc_iter_count

    disc_type = 'x'  # 'x' or 'z' or 'xz'

    # Dimension Parameters
    batch_size = 256
    seed_batch_size = 256

    logit_x_batch_size = 16
    logit_z_batch_size = 16

    # input_size = 2
    z_size = 100

    # Distribution params
    z_bounds = 10.
    cor = 0.6

    # Learning Parameters
    lr_autoencoder = 0.0003
    lr_decoder = 0.0003
    lr_disc = 0.0003

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']

    # model = 'bcgan'
    exp_name = 'mnist_exp_1'

    # dataloader = 'four_gaussian_sym'
    dataloader = 'mnist'

    n_child_nodes = 2

    child_iter = 50

    input_channel = 1
    input_height = 28
    input_width = 28
