from __future__ import print_function, division, absolute_import
import time

import numpy as np
import cv2

from tqdm import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
import os
from models.fashion.gan import ImgGAN

from paths import Paths
from utils.tr_utils import as_np
from base import hyperparams as base_hp

from base.trainer import BaseTrainer
from base.dataloader import BaseDataLoader
from torch.utils.data import DataLoader

from exp_context import ExperimentContext
import math

import torch as tr
from torch.autograd import Variable

exp_name = ExperimentContext.exp_name


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
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
    tensor = as_np(tensor).transpose([0, 2, 3, 1])
    tensor = ((tensor + 1.0) / 2 * 255).astype(np.uint)
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                      normalize=normalize, scale_each=scale_each)  # type: ndarray
    if (filename is not None):
        cv2.imwrite(filename, ndarr[:, :, [2, 1, 0]])

    ndarr = ndarr.transpose([2, 0, 1])

    return ndarr[None, :, :, :]


def create_folders(folders=[]):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


class GanImgTrainer(BaseTrainer):
    def __init__(self, model, data_loader, hyperparams, train_config, tensorboard_msg=''):
        # type: (BaseGan, BaseDataLoader, base_hp.Hyperparams, TrainConfig, str) -> None

        self.H = hyperparams
        self.train_config = train_config
        self.tensorboard_msg = tensorboard_msg
        self.iter_no = 0
        self.train_discriminator = True

        model.create_params_dir()

        self.recon_dir = '../experiments/' + exp_name + '/images/' + model.name + '/recon/'
        self.gen_dir = '../experiments/' + exp_name + '/images/' + model.name + '/gen/'
        self.real_dir = '../experiments/' + exp_name + '/images/' + model.name + '/real/'
        self.gen_strip = '../experiments/' + exp_name + '/images/'+ model.name + '/strip/'

        create_folders([self.gen_dir + 'train/', self.gen_dir + 'test/', 
                        self.recon_dir + 'train/', self.recon_dir + 'test/', 
                        self.real_dir + 'train/', self.real_dir + 'test/',
                        self.gen_strip + 'train/', self.gen_strip + 'test/'])

        print(os.path.dirname(os.path.realpath(__file__)))

        self.writer = {
            'train': SummaryWriter(model.get_log_writer_path('train')),
            'test': SummaryWriter(model.get_log_writer_path('test'))
        }

        seed_data, seed_labels = data_loader.random_batch('test', self.H.seed_batch_size)
        test_seed = {
            'x': seed_data.cuda(),
            'labels': seed_labels.cuda(),
            'z': model.sample((seed_data.shape[0],))
        }

        seed_data, seed_labels = data_loader.random_batch('train', self.H.seed_batch_size)
        train_seed = {
            'x': seed_data.cuda(),
            'labels': seed_labels.cuda(),
            'z': model.sample((seed_data.shape[0],))
        }

        self.seed_data = {
            'train': train_seed,
            'test': test_seed,
        }

        self.pool = Pool(processes=4)

        super(GanImgTrainer, self).__init__(model, data_loader, self.H.n_iterations)

    def is_console_log_step(self):
        n_step = self.train_config.n_step_console_log
        return n_step > 0 and self.iter_no % n_step == 0

    def is_tboard_log_step(self):
        n_step = self.train_config.n_step_tboard_log
        return n_step > 0 and self.iter_no % n_step == 0

    def is_params_save_step(self):
        n_step = self.train_config.n_step_save_params
        return n_step > 0 and self.iter_no % n_step == 0 and self.iter_no != 0

    def is_validation_step(self):
        n_step = self.train_config.n_step_validation
        return n_step > 0 and self.iter_no % n_step == 0 

    def is_visualization_step(self):
        if self.H.show_visual_while_training:
            if self.iter_no % self.train_config.n_step_visualize == 0:
                return True
        return False


    def log_console(self, metrics):
        print('Test Step', self.iter_no + 1)
        print('%s: step %i:     Disc Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_dis_x']))
        print('%s: step %i:     Gen  Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_gen_x']))
        print('%s: step %i: x_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['d_loss']))
        print('%s: step %i: z_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['g_loss']))
        print()

    def validation(self):
        self.model.eval()
        H = self.H
        model = self.model
        dl = self.data_loader


        x_test = dl.test_data()[:64].cuda()
        z_test = model.sample((x_test.shape[0],))

        metrics = model.compute_metrics(x_test, z_test)

        for tag, value in metrics.items():
            self.writer['test'].add_scalar(self.model.name + '_' + tag, value.item(), self.iter_no)

        # Console Log
        if self.is_console_log_step():
            self.log_console(metrics)

        self.model.train()

    def visualize(self, split):

        self.model.eval()
        model = self.model

        data = self.seed_data[split]
       
        x = data['x']
        z = data['z']

        x_recon = model.reconstruct_x(x)
        x_gen = model.decode(z)

        recon_img = save_image(x_recon, filename = self.recon_dir + split + '/' + str(self.iter_no) + '.png')
        gen_img = save_image(x_gen, filename = self.gen_dir + split + '/' + str(self.iter_no) + '.png')
        real = save_image(x, filename = self.real_dir + split + '/' + str(self.iter_no) + '.png')

        image_tag = '%s-plot' % self.model.name

        self.writer[split].add_image(image_tag + 'gen', gen_img[0], self.iter_no)
        self.writer[split].add_image(image_tag + 'recon', recon_img[0], self.iter_no)
        self.writer[split].add_image(image_tag + 'real', real[0], self.iter_no)

        model.train()

    def image_strip(self, split, fixed = False):
        self.model.eval()

        model = self.model
        H = self.H
    
        data = self.data_loader.data[split]

        x1 = tr.Tensor(data[6]).cuda()
        enc1 = model.encoder(x1.unsqueeze(0))
        if fixed:
            np.random.seed(10)
        z1 = model.sample((1,)).cpu().numpy()

        x2 = tr.Tensor(data[9]).cuda()
        enc2 = model.encoder(x2.unsqueeze(0))
        if fixed:
            np.random.seed(80)
        z2 = model.sample((1,)).cpu().numpy()        

        x_real_interpolated = np.asarray([(c*x1.cpu().numpy() + (100-c)*x2.cpu().numpy())/100.0 for c in range(0, 100, 5)])
        enc_interpolated = [(c*enc1.detach().cpu().numpy() + (100-c)*enc2.detach().cpu().numpy())/100.0 for c in range(0, 100, 5)]
        enc_interpolated = tr.Tensor(enc_interpolated).squeeze(1).cuda()
        z_interpolated = [(c*z1 + (100-c)*z2)/100.0 for c in range(0, 100, 5)]
        z_interpolated = tr.Tensor(z_interpolated).squeeze(1).cuda()

        x_enc_interpolated = model.generator(enc_interpolated)
        x_z_interpolated = model.generator(z_interpolated)

        recon_strip = save_image(x_enc_interpolated)
        real_strip = save_image(x_real_interpolated)
        gen_strip = save_image(x_z_interpolated)

        self.writer[split].add_image('recon_strip', recon_strip[0], self.iter_no)
        self.writer[split].add_image('real_strip', real_strip[0], self.iter_no)
        self.writer[split].add_image('gen_strip', gen_strip[0], self.iter_no)

        model.train()

    def full_train_step(self, visualize=True, validation=True, save_params=True):
        dl = self.data_loader
        model = self.model
        H = self.H
            
        x_train, _, _ = dl.next_batch('train')
        x_train = x_train.cuda()

        z_train = model.sample((x_train.shape[0],))

        valid = Variable(tr.Tensor(x_train.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
        fake = Variable(tr.Tensor(x_train.shape[0], 1).fill_(0.0), requires_grad=False).cuda()
        
        encoded_imgs = model.encoder(x_train)
        decoded_imgs = model.generator(encoded_imgs)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        noise_prop = 0.05 # Randomly flip 5% of labels

        # Prepare labels for real data
        true_labels = np.ones((x_train.shape[0], 1)) - np.random.uniform(low=0.0, high=0.1, size=(x_train.shape[0], 1))
        flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop*len(true_labels)))
        true_labels[flipped_idx] = 1 - true_labels[flipped_idx]

        # Prepare labels for generated data
        gene_labels = np.zeros((x_train.shape[0], 1)) + np.random.uniform(low=0.0, high=0.1, size=(x_train.shape[0], 1))
        flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
        gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
        
        true_labels = Variable(tr.Tensor(true_labels), requires_grad=False).cuda()
        gene_labels = Variable(tr.Tensor(gene_labels), requires_grad=False).cuda()

        if self.train_discriminator:

            model.optimizer_D.zero_grad()
            real_loss = model.adversarial_loss(model.discriminator(z_train), true_labels)
            real_loss.backward(retain_graph = True)
            model.optimizer_D.step()
            
            model.optimizer_D.zero_grad()
            fake_loss = model.adversarial_loss(model.discriminator(encoded_imgs), gene_labels)
            fake_loss.backward(retain_graph = True)
            model.optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        model.optimizer_G.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        g_loss =    0.01 * model.adversarial_loss(model.discriminator(encoded_imgs), valid) + \
                    0.99 * model.pixelwise_loss(decoded_imgs, x_train)

        g_loss.backward()
        model.optimizer_G.step()

        # Console Log
        if self.is_console_log_step():
            metrics = model.compute_metrics(x_train, z_train)

            print('============================================================')
            print('Train Step', self.iter_no + 1)
            print('%s: step %i:     Disc Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_dis_x'].item()))
            print('%s: step %i:     Gen  Acc: %.3f' % (exp_name, self.iter_no, metrics['accuracy_gen_x'].item()))
            print('%s: step %i: x_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['d_loss'].item()))
            print('%s: step %i: z_recon Loss: %.3f' % (exp_name, self.iter_no, metrics['g_loss'].item()))
            print('------------------------------------------------------------')
        
        # Tensorboard Log
        if self.is_tboard_log_step():
            metrics = model.compute_metrics(x_train, z_train)
            for tag, value in metrics.items():
                self.writer['train'].add_scalar(self.model.name + '_' + tag, value.item(), self.iter_no)
            
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

        # # Visualization
        if visualize and self.is_visualization_step():
            self.visualize('train')
            self.visualize('test')
            self.image_strip('train')
            self.image_strip('test')


    def resume(self, dir_name, label, iter_no, n_iterations=None):
        self.iter_no = iter_no
        self.model.load_params(dir_name, label, iter_no)
        self.train(n_iterations)

    def train(self, n_iterations=None, enable_tqdm=True, *args, **kwargs):
        dl = self.data_loader
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

        