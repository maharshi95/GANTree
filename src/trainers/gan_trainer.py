from __future__ import print_function, division

import json
from collections import namedtuple
from multiprocessing import Pool
import time
import torch as tr
import matplotlib

from models.toy.nets import ToyGAN
from utils.decorators import numpy_output

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import paths
from utils import viz_utils
from base import hyperparams as base_hp
from base.model import BaseGan
from base.trainer import BaseTrainer
from base.dataloader import BaseDataLoader
from tensorboardX import SummaryWriter
import imageio as im
import numpy as np

from exp_context import ExperimentContext

exp_name = ExperimentContext.exp_name


def get_x_plots_data(model, x_input):
    _, x_real_true, x_real_false = model.discriminate(x_input)

    z_real_true = model.encode(x_real_true, transform=False)
    z_real_false = model.encode(x_real_false, transform=False)

    zt_real_true = model.encode(x_real_true)
    zt_real_false = model.encode(x_real_false)

    x_recon = model.reconstruct_x(x_input)
    _, x_recon_true, x_recon_false = model.discriminate(x_recon)

    z_recon_true = model.encode(x_recon_true, transform=False)
    z_recon_false = model.encode(x_recon_false, transform=False)

    zt_recon_true = model.encode(x_recon_true)
    zt_recon_false = model.encode(x_recon_false)

    return [
        (x_real_true, x_real_false),
        (z_real_true, z_real_false),
        (zt_real_true, zt_real_false),
        (x_recon_true, x_recon_false),
        (z_recon_true, z_recon_false),
        (zt_recon_true, zt_recon_false),
    ]


@numpy_output
def get_z_plots_data(model, z_input):
    x_input = model.decode(z_input)
    x_plots = get_x_plots_data(model, x_input)
    return [z_input] + x_plots[:-1]


@numpy_output
def get_labelled_plots(model, x_input, labels):
    z = model.encode(x_input, transform=False)
    zt = model.encode(x_input)

    x_recon = model.decode(zt)

    z_recon = model.encode(x_recon, transform=False)
    zt_recon = model.encode(x_recon)

    return x_input, z, zt, x_recon, z_recon, zt_recon, labels


def get_plot_data(model, test_seed):
    x_plots_row1 = get_x_plots_data(model, test_seed['x'])
    z_plots_row2 = get_z_plots_data(model, test_seed['z'])
    x_plots_row3 = get_labelled_plots(model, test_seed['x'], test_seed['l'])
    x_plots_row4 = get_x_plots_data(model, test_seed['x_full'])

    plots_data = (x_plots_row1, z_plots_row2, x_plots_row3, x_plots_row4)
    return plots_data


def generate_and_save_image(plots_data, iter_no, scatter_size=0.5):
    print('------------------------------------------------------------')
    print('%s: step %i: started generation' % (exp_name, iter_no))

    figure = viz_utils.get_figure(plots_data, scatter_size)
    print('%s: step %i: got figure' % (exp_name, iter_no))

    figure_name = 'plots-iter-%d.png' % iter_no
    figure_path = paths.get_result_path(figure_name)
    figure.savefig(figure_path)
    plt.close(figure)

    tic_img_save = time.time()
    img = np.array(im.imread(figure_path), dtype=np.uint8)
    img = img[:, :, :-1]
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    print('%s: step %i: visualization saved' % (exp_name, iter_no))
    print('------------------------------------------------------------')
    return img, iter_no


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
        # type: (BaseGan, BaseDataLoader, base_hp.Hyperparams, TrainConfig) -> None

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

        seed_data, seed_labels = data_loader.random_batch('test', self.H.seed_batch_size)
        self.test_seed = {
            'x': seed_data,
            'l': seed_labels,
            'z': model.sample((seed_data.shape[0],)),
            'x_full': data_loader.get_full_space(bounds=4.0)
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

        iter_time_start = time.time()

        x_test, _ = dl.next_batch('test')
        z_test = model.sample((x_test.shape[0],))
        metrics = model.compute_metrics(x_test, z_test)
        g_acc, d_acc = metrics['accuracy_gen'], metrics['accuracy_disc']

        # Tensorboard Log
        if self.is_tboard_log_step():
            for tag, value in metrics.items():
                self.writer['test'].add_scalar(tag, value, self.iter_no)

        # Console Log
        if self.is_console_log_step():
            print('Test Step', self.iter_no + 1)
            print('%s: step %i:     Disc Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_disc']))
            print('%s: step %i:     Gen  Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_gen']))
            print('%s: step %i: x_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_x_recon']))
            print('%s: step %i: z_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_z_recon']))
            print()

            iter_time_end = time.time()

            print('Test Iter Time: %.4f' % (iter_time_end - iter_time_start))
            print('------------------------------------------------------------')

    def train_step(self, x_train, z_train):
        model = self.model
        H = self.H

        if H.train_autoencoder:
            # model.step_train_encoder(x_train, z_train)
            # model.step_train_decoder(z_train)
            model.step_train_autoencoder(x_train, z_train)

        if self.train_generator:
            self.n_iter_gen += 1
            if H.train_generator_adv:
                model.step_train_generator(x_train, z_train)
        else:
            self.n_iter_disc += 1
            model.step_train_discriminator(x_train, z_train)

    def train(self):
        pool = Pool(processes=4)

        dl = self.data_loader
        model = self.model  # type: ToyGAN
        H = self.H

        while self.iter_no < self.n_iterations:
            self.iter_no += 1

            iter_time_start = time.time()

            x_train, _ = dl.next_batch('train')
            z_train = model.sample((H.batch_size,))

            self.train_step(x_train, z_train)

            # Train Losses Computation
            metrics = model.compute_metrics(x_train, z_train)
            g_acc, d_acc = metrics['accuracy_gen'], metrics['accuracy_disc']

            # Console Log
            if self.is_console_log_step():
                print('============================================================')
                print('Train Step', self.iter_no + 1)
                print('%s: step %i:     Disc Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_disc'].item()))
                print('%s: step %i:     Gen  Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_gen'].item()))
                print('%s: step %i: x_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_x_recon'].item()))
                print('%s: step %i: z_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_z_recon'].item()))
                print('------------------------------------------------------------')

            # Tensorboard Log
            if self.is_tboard_log_step():
                for tag, value in metrics.items():
                    self.writer['train'].add_scalar(tag, value.item(), self.iter_no)

            # Validation Computations
            if self.is_validation_step():
                self.validation()

            # Weights Saving
            if self.is_params_save_step():
                tic_save = time.time()
                model.save_params(dir_name='iter', weight_label='iter', iter_no=self.iter_no)
                tac_save = time.time()
                save_time = tac_save - tic_save
                if self.is_console_log_step():
                    print('Param Save Time: %.4f' % (save_time))
                    print('------------------------------------------------------------')

            # Visualization
            if self.is_visualization_step():
                tic_viz = time.time()

                tic_data_prep = time.time()
                plots_data = get_plot_data(self.model, self.test_seed)
                tac_data_prep = time.time()
                time_data_prep = tac_data_prep - tic_data_prep

                writer = self.writer['test']

                def callback(out):
                    img, iter_no = out
                    image_tag = '%s_plot' % self.model.name
                    writer.add_image(image_tag, img, iter_no)

                args = (plots_data, self.iter_no)
                pool.apply_async(generate_and_save_image, args, callback=callback)

                tac_viz = time.time()

                time_viz = tac_viz - tic_viz
                if self.is_console_log_step():
                    print('Data Prep     Time: %.4f' % (time_data_prep))
                    print('Visualization Time: %.4f' % (time_viz))
                    print('------------------------------------------------------------')

            # Switch Training Networks - Gen | Disc
            self.switch_train_mode(g_acc, d_acc)

            iter_time_end = time.time()
            if self.is_console_log_step():
                print('Total Iter Time: %.4f' % (iter_time_end - iter_time_start))
                print('============================================================')
                print()
