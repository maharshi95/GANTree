from base import hyperparams
import torch as tr

class Hyperparams(hyperparams.Hyperparams):  # change

    dtype = float

    n_iterations = 26000

    start_tensorboard = True

    # Dimension Parameters
    batch_size = 64
    seed_batch_size = 64

    z_dim = 8
    dmu = 17
    cor = 0.0

    threshold = 2.5

    save_node = False

    root_gan_iters = 40000
    phase1_epochs = 13
    phase2_iters = 20000

    n_step_tboard_log = 100
    n_step_console_log = -1
    n_step_validation = 100
    n_step_save_params = 10000
    n_step_visualize = 500

    # Learning Parameters
    lr = 0.0002

    b1 = 0.5
    b2 = 0.999
    # n_epochs = 60

    img_size = 64
    channel = 1

    img_shape = (channel, img_size, img_size)

    epsilon = 1e-15

    show_visual_while_training = True

    dataloader = 'mnist'
    no_of_classes = 10

    n_child_nodes = 2

    z_bounds = 10.0

    @classmethod
    def z_means(cls):
        return tr.zeros(cls.z_dim)

    @classmethod
    def z_cov(cls, sign='0'):
        cov = tr.eye(cls.z_dim)
        cor = {
            '+': cls.cor,
            '-': -cls.cor,
            '0': 0.0
        }[sign]
        cov[0, 1] = cov[1, 0] = cor
        return cov
