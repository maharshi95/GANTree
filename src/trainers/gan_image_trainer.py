from __future__ import print_function, division, absolute_import
import time
import numpy as np
from sklearn import preprocessing as prep
from numpy import linalg as LA
import imageio as im

from tqdm import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
import os
from configs import TrainConfig
from models.toy.gan import ToyGAN
from utils.decorators import numpy_output

from paths import Paths
from utils import viz_utils
from utils.tr_utils import as_np
from base import hyperparams as base_hp
from base.model import BaseGan
from base.trainer import BaseTrainer
from base.dataloader import BaseDataLoader

from exp_context import ExperimentContext
import math
from PIL import Image

exp_name = ExperimentContext.exp_name


#
# def (z):
#
#     #  z.shape[0]:batchsize
#     #  z.shape[1]:z_size
#     z = as_np(z)
#     data_scaled = prep.scale(z)
#
#     w, v = LA.eig(np.cov(data_scaled, rowvar=False))
#     _main_axes = v[:, :2]
#     projected_data = np.dot(data_scaled, _main_axes)
#
#     return projected_data
#


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


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.float16)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h + h_width, w:w + w_width] = tensor[k]
            k = k + 1
    return grid


def save_image(tensor, filename=None, nrow=8, padding=2,
               normalize=False, scale_each=False):
    tensor = tensor[:, 0, :, :, None]
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                      normalize=normalize, scale_each=scale_each)
    # img = Image.fromarray(ndarr)

    h, w, c = ndarr.shape
    ndarr = ndarr.transpose([2, 0, 1])
    return ndarr[None, :, :, :]
    # img.save(filename)


class GanImgTrainer(BaseTrainer):
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

        self.recon_dir = '../experiments/' + exp_name + '/images/recon/'
        self.gen_dir = '../experiments/' + exp_name + '/images/gen/'
        if not os.path.exists(self.recon_dir):
            os.makedirs(self.recon_dir)

        if not os.path.exists(self.gen_dir):
            os.makedirs(self.gen_dir)

        print(os.path.dirname(os.path.realpath(__file__)))

        self.writer = {
            'train': SummaryWriter(model.get_log_writer_path('train')),
            'test': SummaryWriter(model.get_log_writer_path('test')),
        }

        # bounds = 6

        seed_data, seed_labels = data_loader.random_batch('test', self.H.seed_batch_size)
        test_seed = {
            'x': seed_data,
            'l': seed_labels,
            'z': model.sample((seed_data.shape[0],)),
        }

        seed_data, seed_labels = data_loader.random_batch('test', self.H.seed_batch_size)
        self.fixed_seed = {
            'x': seed_data,
            'l': seed_labels,
            'z': model.sample((seed_data.shape[0],)),

        }

        seed_data, seed_labels = data_loader.random_batch('train', self.H.seed_batch_size)
        train_seed = {
            'x': seed_data,
            'l': seed_labels,
            'z': model.sample((seed_data.shape[0],)),
        }

        self.seed_data = {
            'train': train_seed,
            'test': test_seed,
        }

        self.pool = Pool(processes=4)

        super(GanImgTrainer, self).__init__(model, data_loader, self.H.n_iterations)

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
        self.model.eval()
        H = self.H
        model = self.model
        dl = self.data_loader

        iter_time_start = time.time()

        x_test, _ = dl.next_batch('test')
        z_test = model.sample((x_test.shape[0],))
        metrics = model.compute_metrics(x_test, z_test, True)
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
        self.model.train()

    def visualize(self, split):
        self.model.eval()
        tic_viz = time.time()
        tic_data_prep = time.time()
        recon, gen, real = self.save_img(self.seed_data[split])
        # print(split, 'recon', recon.shape, recon.min(), recon.max())
        # print(split, 'gen', gen.shape, gen.min(), gen.max())
        # print(split, 'real', real.shape, real.min(), real.max())
        tac_data_prep = time.time()
        time_data_prep = tac_data_prep - tic_data_prep

        writer = self.writer[split]
        image_tag = '%s-plot' % self.model.name
        iter_no = self.iter_no

        writer.add_image(image_tag + '-recon', recon, iter_no)
        writer.add_image(image_tag + '-gen', gen, iter_no)
        writer.add_image(image_tag + '-real', real, iter_no)

        self.model.train()

    def train_step_ae(self, x_train, z_train):
        if self.H.train_autoencoder:
            # model.step_train_encoder(x_train, z_train)
            # model.step_train_decoder(z_train)
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

    def save_img(self, test_seed=None):
        # test_seed = self.fixed_seed if test_seed is None else test_seed
        x = test_seed['x']
        z = test_seed['z']

        x_recon = self.model.reconstruct_x(x)
        x_gen = self.model.decode(z)
        # shape:[1,c,h,w]
        recon_img = save_image(x_recon, self.recon_dir + str(self.iter_no) + '.png')
        gen_img = save_image(x_gen, self.gen_dir + str(self.iter_no) + '.png')
        real = save_image(x)

        return recon_img, gen_img, real

    def full_train_step(self, visualize=True, validation=True, save_params=True):
        dl = self.data_loader
        model = self.model  # type: ImgGAN
        H = self.H

        iter_time_start = time.time()

        x_train, _ = dl.next_batch('train')
        z_train = model.sample((H.batch_size,))
        self.train_step_ae(x_train, z_train)
        self.train_step_ad(x_train, z_train)
        # net_id = 1 if self.train_generator else 0
        # self.writer['train'].add_scalar('is_gen', net_id, self.iter_no)

        # Train Losses Computation
        self.model.eval()
        x_train_batch = self.seed_data['train']['x']
        z_train_batch = self.seed_data['train']['z']

        metrics = model.compute_metrics(x_train_batch, z_train_batch)
        self.model.train()

        g_acc, d_acc = metrics['accuracy_gen_x'], metrics['accuracy_dis_x']

        # # Console Log
        # if self.is_console_log_step():
        #     print('============================================================')
        #     print('Train Step', self.iter_no + 1)
        #     print('%s: step %i:     Disc Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_dis_x'].item()))
        #     print('%s: step %i:     Gen  Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_gen_x'].item()))
        #     print('%s: step %i: x_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_x_recon'].item()))
        #     print('%s: step %i: z_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['loss_z_recon'].item()))
        #     print('------------------------------------------------------------')
        #
        # Tensorboard Log
        if self.is_tboard_log_step():
            for tag, value in metrics.items():
                self.writer['train'].add_scalar(tag, value.item(), self.iter_no)
            self.writer['train'].add_scalar('switch_train_mode', int(self.train_generator), self.iter_no)

        #
        # Validation Computations
        if validation and self.is_validation_step():
            self.validation()
        #
        # Weights Saving
        if save_params and self.is_params_save_step():
            tic_save = time.time()
            model.save_params(dir_name='iter', weight_label='iter', iter_no=self.iter_no)
            tac_save = time.time()
            save_time = tac_save - tic_save
            if self.is_console_log_step():
                print('Param Save Time: %.4f' % (save_time))
                print('------------------------------------------------------------')

        # # Visualization
        if visualize and self.is_visualization_step():
            # previous_backend = plt.get_backend()
            # plt.switch_backend('Agg')
            self.visualize('train')
            self.visualize('test')
            # plt.switch_backend(previous_backend)

        # Switch Training Networks - Gen | Disc
        self.switch_train_mode(g_acc, d_acc)

        # iter_time_end = time.time()
        # if self.is_console_log_step():
        #     print('Total Iter Time: %.4f' % (iter_time_end - iter_time_start))
        #     if self.tensorboard_msg:
        #         print('------------------------------------------------------------')
        #         print(self.tensorboard_msg)
        #     print('============================================================')
        #     print()

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
