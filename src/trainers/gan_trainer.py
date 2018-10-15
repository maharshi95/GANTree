from __future__ import print_function, division
import torch as tr

from base.model import BaseGan
from base.trainer import BaseTrainer
from base.dataloader import BaseDataLoader


class GanTrainer(BaseTrainer):
    def __init__(self, model, data_loader, n_iterations):
        # type: (BaseGan, BaseDataLoader, int) -> None
        super(GanTrainer, self).__init__(model, data_loader, n_iterations)

    def train(self):
        dl = self.data_loader
        model = self.model

        for iter_no in range(self.n_iterations):
            x_train = tr.Tensor(dl.next_batch('train'))
            z_train = tr.Tensor(dl.next_batch('train'))

            if iter_no % 20 < 15:
                c_loss = model.step_train_autoencoder(x_train, z_train)
                g_loss = model.step_train_generator(z_train)
            else:
                # pass
                d_loss = model.step_train_discriminator(x_train, z_train)
            g_acc, d_acc = model.get_accuracies(x_train, z_train)

            x_test = dl.next_batch('test')

            if iter_no % 100 == 99:
                print('Step', iter_no + 1)
                print('Gen  Accuracy:', g_acc.item())
                print('Disc Accuracy:', d_acc.item())


"""
modify later
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from dataloaders.multi_normal import FourSymGaussiansDataLoader
import numpy as np
from hyperparams.toy import bcgan_2d
from exp_context import ExperimentContext

ExperimentContext.set_context(bcgan_2d)  # hyperparams name

from models.toy.nets import ToyGAN
from utils import viz_utils




from tf import paths

from torch.autograd import variable
dl = FourSymGaussiansDataLoader()
H = ExperimentContext.get_hyperparams()



max_epochs = 20000

n_step_console_log = 500
n_step_tboard_log = 10
n_step_validation = 50
n_step_iter_save = 5000
n_step_visualize = 500
n_step_generator = 10

en_loss_history = []
de_loss_history = []
di_loss_history = []
d_acc_history = []
g_acc_history = []
gen_loss_history = []

model = ToyGAN('gan')
iter_no = 0
x_train, x_test = dl.get_data()
print('Train Test Data loaded...')

while iter_no < max_epochs:
    iter_no += 1

    z_train = dl.get_z_dist(x_train.shape[0], dist_type=H.z_dist_type)
    z_test = dl.get_z_dist(x_test.shape[0], dist_type=H.z_dist_type)

    train_inputs = x_train, z_train
    test_inputs = x_test, z_test

    model.step_train_autoencoder(tr.FloatTensor(x_train), tr.FloatTensor(z_train))

    if (iter_no % n_step_generator) == 0:
        if H.train_generator_adv:
            model.step_train_generator(tr.FloatTensor(z_train))
    else:
        model.step_train_discriminator(tr.FloatTensor(x_train), tr.FloatTensor(z_train))

    g_acc, d_acc = model.get_accuracies(tr.FloatTensor(x_train), tr.FloatTensor(z_train))

    x_test = dl.next_batch('test')

    if iter_no % 10 == 9:
        print('Step', iter_no + 1)
        print('Gen  Accuracy:', g_acc.item())
        print('Disc Accuracy:', d_acc.item())


    if H.show_visual_while_training and (iter_no % n_step_visualize == 0 or (iter_no < n_step_visualize and iter_no % 200 == 0)):
        def get_x_plots_data(x_input):
            _, x_real_true, x_real_false = model.discriminate(x_input)

            z_real_true = model.encode(x_real_true)
            z_real_false = model.encode(x_real_false)

            x_recon = model.reconstruct_x(x_input)
            _, x_recon_true, x_recon_false = model.discriminate(x_recon)

            z_recon_true = model.encode(x_recon_true)
            z_recon_false = model.encode(x_recon_false)

            return [
                (x_real_true.data.numpy(), x_real_false.numpy()),
                (z_real_true.numpy(), z_real_false.numpy()),
                (x_recon_true.numpy(), x_recon_false.numpy()),
                (z_recon_true.numpy(), z_recon_false.numpy())
            ]


        def get_z_plots_data(z_input):
            z_input = tr.from_numpy(z_input)

            x_input = model.decode(z_input)
            x_plots = get_x_plots_data(x_input)
            return [z_input.numpy()] + x_plots[:-1]


        x_full = dl.get_full_space() # np

        x_plots_row1 = get_x_plots_data(x_test)
        z_plots_row2 = get_z_plots_data(z_test)   # flag!!!!!!!!!!
        x_plots_row3 = get_x_plots_data(x_full)
        plots_data = (x_plots_row1, z_plots_row2, x_plots_row3)
        figure = viz_utils.get_figure(plots_data)
        figure_name = 'plots-iter-%d.png' % iter_no
        figure_path = paths.get_result_path(figure_name)
        figure.savefig(figure_path)
        plt.close(figure)
        img = plt.imread(figure_path)
        # model.log_image('test', img, iter_no)

    # iter_time_end = time.time()
    #
    # if iter_no % n_step_console_log == 0:
    #     print('Single Iter Time: %.4f' % (iter_time_end - iter_time_start))
