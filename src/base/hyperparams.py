import torch as tr


class Hyperparams:
    """
    Base Hyperparams class.
    It uses base version of bcgan with 1D x space and z space
    """
    dtype = float

    # Trainer parameters:
    n_iterations = 10000

    show_visual_while_training = True
    train_generator_adv = True
    train_autoencoder = True

    train_batch_logits = True
    train_sample_logits = True

    start_tensorboard = True

    circular_bounds = False

    gen_iter_count = 40
    disc_iter_count = 40
    step_ratio = gen_iter_count, disc_iter_count

    disc_type = 'x'  # 'x' or 'z' or 'xz'

    # Dimension Parameters
    batch_size = 1024
    seed_batch_size = 1024

    logit_x_batch_size = 4
    logit_z_batch_size = 1

    input_size = 2
    z_size = 100

    # Distribution params
    z_bounds = 10.0
    cor = 0.6

    # Learning Parameters
    lr_autoencoder = 0.0003
    lr_decoder = 0.0003
    lr_disc = 0.0003

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']

    model = 'bcgan'
    exp_name = 'trial_with_gmms'

    # dataloader = 'four_gaussian_sym'
    dataloader = 'nine_gaussian'

    n_child_nodes = 2

    child_iter = 50

    input_channel = 1
    input_height = 32
    input_width = 32

    @classmethod
    def z_means(cls):
        return tr.zeros(cls.z_size)

    @classmethod
    def z_cov(cls, sign='0'):
        cov = tr.eye(cls.z_size)
        cor = {
            '+': cls.cor,
            '-': -cls.cor,
            '0': 0.0
        }[sign]
        cov[0, 1] = cov[1, 0] = cor
        return cov
