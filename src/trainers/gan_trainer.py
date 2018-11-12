from __future__ import print_function, division, absolute_import
import time
import numpy as np
import imageio as im

from tqdm import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from configs import TrainConfig
from models.toy.gan import ToyGAN
from utils.decorators import numpy_output

from paths import Paths
from utils import viz_utils

from base import hyperparams as base_hp
from base.model import BaseGan
from base.trainer import BaseTrainer
from base.dataloader import BaseDataLoader

from exp_context import ExperimentContext

exp_name = ExperimentContext.exp_name


def get_x_plots_data(model, x_input):
    _, x_real_true, x_real_false = model.discriminate(x_input)

    z_real_true = model.encode(x_real_true, transform=False)
    z_real_false = model.encode(x_real_false, transform=False)

    zt_real_true = model.encode(x_real_true, transform=True)
    zt_real_false = model.encode(x_real_false, transform=True)

    x_recon = model.reconstruct_x(x_input)
    _, x_recon_true, x_recon_false = model.discriminate(x_recon)

    z_recon_true = model.encode(x_recon_true, transform=False)
    z_recon_false = model.encode(x_recon_false, transform=False)

    zt_recon_true = model.encode(x_recon_true, transform=True)
    zt_recon_false = model.encode(x_recon_false, transform=True)

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
    zt = model.encode(x_input, transform=True)

    x_recon = model.decode(zt)

    z_recon = model.encode(x_recon, transform=False)
    zt_recon = model.encode(x_recon, transform=True)

    # return x_input, z, zt, x_recon, z_recon, zt_recon, labels
    return x_input, z, zt, x_recon, z, zt, labels


def get_plot_data(model, test_seed):
    x_plots_row1 = get_x_plots_data(model, test_seed['x'])
    z_plots_row2 = get_z_plots_data(model, test_seed['z'])
    x_plots_row3 = get_labelled_plots(model, test_seed['x'], test_seed['l'])
    x_plots_row4 = get_x_plots_data(model, test_seed['x_full'])

    z_data = x_plots_row3[1]
    zt_data = x_plots_row3[2]

    plots_data = (x_plots_row1, z_plots_row2, x_plots_row3, x_plots_row4, (z_data, zt_data))
    return plots_data


def generate_and_save_image(plots_data, iter_no, image_label, scatter_size=0.5, log=False):
    if log:
        print('------------------------------------------------------------')
        print('%s: step %i: started generation' % (exp_name, iter_no))

    figure = viz_utils.get_figure(plots_data, scatter_size)
    if log:
        print('%s: step %i: got figure' % (exp_name, iter_no))

    figure_name = '%s-%05d.png' % (image_label, iter_no)
    figure_path = Paths.get_result_path(figure_name)
    figure.savefig(figure_path)
    plt.close(figure)

    img = np.array(im.imread(figure_path), dtype=np.uint8)
    img = img[:, :, :-1]
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    if log:
        print('%s: step %i: visualization saved' % (exp_name, iter_no))
        print('------------------------------------------------------------')
    return img, iter_no


class GanTrainer(BaseTrainer):
    def __init__(self, model, data_loader, hyperparams, train_config, tensorboard_msg='', auto_encoder_dl=None):
        # type: (BaseGan, BaseDataLoader, base_hp.Hyperparams, TrainConfig, str) -> None

        self.H = hyperparams
        self.iter_no = 1
        self.n_iter_gen = 0
        self.n_iter_disc = 0
        self.n_step_gen, self.n_step_disc = self.H.step_ratio
        self.train_generator = True
        self.train_config = train_config
        self.tensorboard_msg = tensorboard_msg
        self.auto_encoder_dl = auto_encoder_dl

        model.create_params_dir()

        self.writer = {
            'train': SummaryWriter(model.get_log_writer_path('train')),
            'test': SummaryWriter(model.get_log_writer_path('test')),
        }

        bounds = 6

        seed_data, seed_labels = data_loader.random_batch('test', self.H.seed_batch_size)
        test_seed = {
            'x': seed_data,
            'l': seed_labels,
            'z': model.sample((seed_data.shape[0],)),
            'x_full': data_loader.get_full_space(bounds=6.0, n_samples=1024)
        }

        seed_data, seed_labels = data_loader.random_batch('train', self.H.seed_batch_size)
        train_seed = {
            'x': seed_data,
            'l': seed_labels,
            'z': model.sample((seed_data.shape[0],)),
            'x_full': data_loader.get_full_space(bounds=6.0, n_samples=1024)
        }

        self.seed_data = {
            'train': train_seed,
            'test': test_seed
        }

        self.pool = Pool(processes=4)

        super(GanTrainer, self).__init__(model, data_loader, self.H.n_iterations)

    def gen_train_limit_reached(self, gen_accuracy):
        return self.n_iter_gen == self.H.gen_iter_count or gen_accuracy >= 80

    def disc_train_limit_reached(self, disc_accuracy):
        return self.n_iter_disc == self.H.disc_iter_count or disc_accuracy >= 90

    # Check Functions for various operations
    def is_console_log_step(self):
        n_step = self.train_config.n_step_console_log
        return n_step > 0 and self.iter_no % n_step == 0

    def is_tboard_log_step(self):
        n_step = self.train_config.n_step_tboard_log
        return n_step > 0 and self.iter_no % n_step == 0

    def is_params_save_step(self):
        n_step = self.train_config.n_step_save_params
        return n_step > 0 and self.iter_no % n_step == 0

    def is_validation_step(self):
        n_step = self.train_config.n_step_validation
        return n_step > 0 and self.iter_no % n_step == 0

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

    def log_console(self, metrics):
        print('Test Step', self.iter_no + 1)
        print('%s: step %i:     Disc Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_dis_x']))
        print('%s: step %i:     Gen  Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_gen_x']))
        print('%s: step %i: x_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_x_recon']))
        print('%s: step %i: z_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_z_recon']))
        print()

    def validation(self):
        H = self.H
        model = self.model
        dl = self.data_loader

        iter_time_start = time.time()

        x_test, _ = dl.next_batch('test')
        z_test = model.sample((x_test.shape[0],))
        metrics = model.compute_metrics(x_test, z_test)
        g_acc, d_acc = metrics['accuracy_gen_x'], metrics['accuracy_dis_x']

        # Tensorboard Log
        if self.is_tboard_log_step():
            for tag, value in metrics.items():
                self.writer['test'].add_scalar(tag, value, self.iter_no)

        # Console Log
        if self.is_console_log_step():
            self.log_console(metrics)
            iter_time_end = time.time()

            print('Test Iter Time: %.4f' % (iter_time_end - iter_time_start))
            print('------------------------------------------------------------')

    def visualize(self, split):
        tic_viz = time.time()

        tic_data_prep = time.time()
        plots_data = get_plot_data(self.model, self.seed_data[split])
        tac_data_prep = time.time()
        time_data_prep = tac_data_prep - tic_data_prep

        writer = self.writer[split]
        image_tag = '%s-plot' % self.model.name

        def callback(out):
            img, iter_no = out
            writer.add_image(image_tag, img, iter_no)

        args = (plots_data, self.iter_no)
        kwargs = {
            'image_label': image_tag,
            'scatter_size': 0.5,
            'log': self.is_console_log_step()
        }
        # out = generate_and_save_image(*args, **kwargs)
        # callback(out)
        self.pool.apply_async(generate_and_save_image, args, kwargs, callback=callback)

        tac_viz = time.time()

        time_viz = tac_viz - tic_viz
        if self.is_console_log_step():
            print('Data Prep     Time: %.4f' % (time_data_prep))
            print('Visualization Time: %.4f' % (time_viz))
            print('------------------------------------------------------------')

    def train_step_ae(self, x_train, z_train):
        if self.H.train_autoencoder:
            self.model.step_train_autoencoder(x_train, z_train)

    def train_step_ad(self, x_train, z_train):
        model = self.model
        H = self.H

        if self.train_generator:
            self.n_iter_gen += 1
            if H.train_generator_adv:
                model.step_train_generator(x_train, z_train)
        else:
            self.n_iter_disc += 1
            model.step_train_discriminator(x_train, z_train)

    def train_step_2(self, x_train, z_train):
        model = self.model

        if self.train_generator:
            self.n_iter_gen += 1
            model.step_train_encoder(x_train, z_train, lam=0.001)
            model.step_train_decoder(x_train, z_train, lam=0.001)
        else:
            self.n_iter_disc += 1
            model.step_train_discriminator(x_train, z_train)

    def full_train_step(self, visualize=True, validation=True, save_params=True):
        dl = self.data_loader
        model = self.model  # type: ToyGAN
        H = self.H

        iter_time_start = time.time()

        x_train, _ = dl.next_batch('train')
        z_train = model.sample((H.batch_size,))

        self.train_step_ae(x_train, z_train)
        self.train_step_ad(x_train, z_train)
        # net_id = 1 if self.train_generator else 0
        # self.writer['train'].add_scalar('is_gen', net_id, self.iter_no)

        # Train Losses Computation
        metrics = model.compute_metrics(x_train, z_train)
        g_acc, d_acc = metrics['accuracy_gen_x'], metrics['accuracy_dis_x']

        # Console Log
        if self.is_console_log_step():
            print('============================================================')
            print('Train Step', self.iter_no + 1)
            print('%s: step %i:     Disc Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_dis_x'].item()))
            print('%s: step %i:     Gen  Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_gen_x'].item()))
            print('%s: step %i: x_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_x_recon'].item()))
            print('%s: step %i: z_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_z_recon'].item()))
            print('------------------------------------------------------------')

        # Tensorboard Log
        if self.is_tboard_log_step():
            for tag, value in metrics.items():
                self.writer['train'].add_scalar(tag, value.item(), self.iter_no)

        # Validation Computations
        if validation and self.is_validation_step():
            self.validation()

        # Weights Saving
        if save_params and self.is_params_save_step():
            tic_save = time.time()
            model.save_params(dir_name='iter', weight_label='iter', iter_no=self.iter_no)
            tac_save = time.time()
            save_time = tac_save - tic_save
            if self.is_console_log_step():
                print('Param Save Time: %.4f' % (save_time))
                print('------------------------------------------------------------')

        # Visualization
        if visualize and self.is_visualization_step():
            # previous_backend = plt.get_backend()
            # plt.switch_backend('Agg')
            self.visualize('train')
            self.visualize('test')
            # plt.switch_backend(previous_backend)

        # Switch Training Networks - Gen | Disc
        self.switch_train_mode(g_acc, d_acc)

        iter_time_end = time.time()
        if self.is_console_log_step():
            print('Total Iter Time: %.4f' % (iter_time_end - iter_time_start))
            if self.tensorboard_msg:
                print('------------------------------------------------------------')
                print(self.tensorboard_msg)
            print('============================================================')
            print()

    def resume(self, dir_name, label, iter_no, n_iterations=None):
        self.iter_no = iter_no
        self.model.load_params(dir_name, label, iter_no)
        self.train(n_iterations)

    def train(self, n_iterations=None, enable_tqdm=True, *args, **kwargs):
        n_iterations = n_iterations or self.n_iterations
        start_iter = self.iter_no
        end_iter = start_iter + n_iterations + 1

        if enable_tqdm:
            with tqdm(total=n_iterations) as pbar:
                for self.iter_no in range(start_iter, end_iter):
                    self.full_train_step(*args, **kwargs)
                    pbar.update(1)


        else:
            for self.iter_no in range(start_iter, end_iter):
                self.full_train_step(*args, **kwargs)
