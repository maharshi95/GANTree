from __future__ import print_function, division
from collections import namedtuple

import time
import torch as tr
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import paths
from utils import viz_utils
from base import hyperparams as base
from base.model import BaseGan
from base.trainer import BaseTrainer
from base.dataloader import BaseDataLoader
from tensorboardX import SummaryWriter
import imageio as im
import numpy as np


def get_x_plots_data(model, x_input):
    _, x_real_true, x_real_false = model.discriminate(x_input)

    z_real_true = model.encode(x_real_true)
    z_real_false = model.encode(x_real_false)

    x_recon = model.reconstruct_x(x_input)
    _, x_recon_true, x_recon_false = model.discriminate(x_recon)

    z_recon_true = model.encode(x_recon_true)
    z_recon_false = model.encode(x_recon_false)

    return [
        (x_real_true.numpy(), x_real_false.numpy()),
        (z_real_true.numpy(), z_real_false.numpy()),
        (x_recon_true.numpy(), x_recon_false.numpy()),
        (z_recon_true.numpy(), z_recon_false.numpy())
    ]


def get_z_plots_data(model, z_input):
    x_input = model.decode(z_input)
    x_plots = get_x_plots_data(model, x_input)
    return [z_input.numpy()] + x_plots[:-1]


TrainConfig = namedtuple(
    'TrainConfig',
    'n_step_tboard_log '
    'n_step_console_log '
    'n_step_validation '
    'n_step_save_params '
    'n_step_visualize'
)


class GanTrainer(BaseTrainer):
    def __init__(self, model, data_loader, hyperparams, train_config):
        # type: (BaseGan, BaseDataLoader, base.Hyperparams, TrainConfig) -> None

        self.H = hyperparams
        self.iter_no = 0
        self.n_iter_gen = 0
        self.n_iter_disc = 0
        self.n_step_gen, self.n_step_disc = self.H.step_ratio
        self.train_generator = True
        self.train_config = train_config

        self.writer = {
            'train': SummaryWriter(paths.log_writer_path('train')),
            'test': SummaryWriter(paths.log_writer_path('test')),
        }

        self.test_seed = {
            'x': data_loader.next_batch('test'),
            'z': data_loader.get_z_dist(self.H.batch_size, dist_type=self.H.z_dist_type, bounds=self.H.z_bounds)
        }

        super(GanTrainer, self).__init__(model, data_loader, self.H.n_iterations)

    def gen_train_limit_reached(self, gen_accuracy):
        return self.n_iter_gen == self.H.gen_iter_count or gen_accuracy >= 70

    def disc_train_limit_reached(self, disc_accuracy):
        return self.n_iter_disc == self.H.disc_iter_count or disc_accuracy >= 95

    # Check Functions for various operations
    def is_console_log_step(self):
        return self.iter_no % self.train_config.n_step_console_log == 0

    def is_tboard_log_step(self):
        return self.iter_no % self.train_config.n_step_tboard_log == 0

    def is_params_save_step(self):
        return self.iter_no % self.train_config.n_step_save_params == 0

    def is_validation_step(self):
        return self.iter_no % self.train_config.n_step_validation == 0

    def is_visualization_step(self):
        if self.H.show_visual_while_training:
            if self.iter_no % self.train_config.n_step_visualize == 0:
                return True
            elif self.iter_no < self.train_config.n_step_visualize:
                if self.iter_no % 200 == 0:
                    return True
        return False

    # Conditional Switch - Training Networks
    def switch_train_mode(self, gen_accuracy, disc_accuracy):
        if self.train_generator:
            if self.gen_train_limit_reached(gen_accuracy):
                self.n_iter_gen = 0
                self.train_generator = False

        if not self.train_generator:
            if self.disc_train_limit_reached(disc_accuracy):
                self.n_iter_disc = 0
                self.train_generator = True

    def validation(self):
        H = self.H
        model = self.model
        dl = self.data_loader

        x_test = dl.next_batch('test')
        z_test = dl.get_z_dist(x_test.shape[0], dist_type=H.z_dist_type, bounds=H.z_bounds)
        metrics = model.compute_metrics(x_test, z_test)
        g_acc, d_acc = metrics['accuracy_gen'], metrics['accuracy_disc']

        print('Test Step', self.iter_no + 1)
        print('Step %i: Disc Acc: %f' % (self.iter_no, metrics['accuracy_disc']))
        print('Step %i: Gen  Acc: %f' % (self.iter_no, metrics['accuracy_gen']))
        print('Step %i: x_recon Loss: %f' % (self.iter_no, metrics['loss_x_recon']))
        print('Step %i: z_recon Loss: %f' % (self.iter_no, metrics['loss_z_recon']))
        print()

        # Tensorboard Log
        if self.is_tboard_log_step():
            for tag, value in metrics.items():
                self.writer['test'].add_scalar(tag, value, self.iter_no)

    def train(self):
        dl = self.data_loader
        model = self.model
        H = self.H

        while self.iter_no < self.n_iterations:
            self.iter_no += 1

            iter_time_start = time.time()

            x_train = dl.next_batch('train')
            z_train = dl.get_z_dist(x_train.shape[0], dist_type=H.z_dist_type, bounds=H.z_bounds)

            if H.train_autoencoder:
                model.step_train_autoencoder(x_train, z_train)

            if self.train_generator:
                self.n_iter_gen += 1
                if H.train_generator_adv:
                    model.step_train_generator(z_train)
            else:
                self.n_iter_disc += 1
                model.step_train_discriminator(x_train, z_train)

            # Train Losses Computation
            metrics = model.compute_metrics(x_train, z_train)
            g_acc, d_acc = metrics['accuracy_gen'], metrics['accuracy_disc']

            # Console Log
            if self.is_console_log_step():
                print('Train Step', self.iter_no + 1)
                print('Gen  Accuracy:', g_acc.item())
                print('Disc Accuracy:', d_acc.item())

                print('Step %i: Disc Acc: %f' % (self.iter_no, metrics['accuracy_disc'].item()))
                print('Step %i: Gen  Acc: %f' % (self.iter_no, metrics['accuracy_gen'].item()))
                print('Step %i: x_recon Loss: %f' % (self.iter_no, metrics['loss_x_recon'].item()))
                print('Step %i: z_recon Loss: %f' % (self.iter_no, metrics['loss_z_recon'].item()))
                print()

            # Tensorboard Log
            if self.is_tboard_log_step():
                for tag, value in metrics.items():
                    self.writer['train'].add_scalar(tag, value.item(), self.iter_no)

            # Validation Computations
            if self.is_validation_step():
                self.validation()

            # Weights Saving
            if self.is_params_save_step():
                model.save_params(dir_name='iter', weight_label='iter', iter_no=self.iter_no)

            # Visualization
            if self.is_visualization_step():
                x_full = dl.get_full_space()

                x_plots_row1 = get_x_plots_data(model, self.test_seed['x'])
                z_plots_row2 = get_z_plots_data(model, self.test_seed['z'])
                x_plots_row3 = get_x_plots_data(model, x_full)

                plots_data = (x_plots_row1, z_plots_row2, x_plots_row3)

                figure = viz_utils.get_figure(plots_data)
                figure_name = 'plots-iter-%d.png' % self.iter_no
                figure_path = paths.get_result_path(figure_name)
                figure.savefig(figure_path)
                plt.close(figure)

                img = np.array(im.imread(figure_path), dtype=np.uint8)
                img = img[:, :, :-1]
                img = img.transpose(2, 0, 1)
                img = np.expand_dims(img, 0)
                # image = tr.from_numpy(img)
                self.writer['test'].add_image('plot_iter', img, self.iter_no)

            # Switch Training Networks
            self.switch_train_mode(g_acc, d_acc)

            iter_time_end = time.time()
            if self.is_console_log_step():
                print('Single Iter Time: %.4f' % (iter_time_end - iter_time_start))



