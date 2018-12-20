from base import hyperparams
import torch as tr

class Hyperparams(hyperparams.Hyperparams):  # change

    dtype = float

    # Trainer parameters:
    n_iterations = 100000

    show_visual_while_training = True
    train_generator_adv = True
    train_autoencoder = True

    train_batch_logits = False
    train_sample_logits = True

    start_tensorboard = True

    circular_bounds = False

    gen_iter_count = 40
    disc_iter_count = 40
    step_ratio = gen_iter_count, disc_iter_count

    disc_type = 'x'  # 'x' or 'z' or 'xz'

    # Dimension Parameters
    batch_size = 128
    seed_batch_size = 64

    logit_x_batch_size = 16
    logit_z_batch_size = 16

    # input_size = 2
    z_size = 100

    # Distribution params
    z_bounds = 10.
    cor = 0.0

    # Learning Parameters
    lr_autoencoder = 0.0003
    lr_decoder = 0.0003
    lr_disc = 0.0003

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']

    # model = 'bcgan'
    exp_name = 'exp_30_facebed'

    # dataloader = 'four_gaussian_sym'
    dataloader = 'facebed'

    n_child_nodes = 2

    child_iter = 50

    input_channel = 3
    input_height = 64
    input_width = 64

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
