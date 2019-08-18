from __future__ import print_function, division
import json, time
from collections import namedtuple
from tqdm import tqdm
from multiprocessing import Pool
import torch as tr
import matplotlib

from dataloaders.custom_loader import CustomDataLoader
from models.toy.gt.gnode import GNode
from models.toy.gan import ToyGAN
from utils.decorators import numpy_output

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from paths import Paths
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


TrainConfig = namedtuple(
    'TrainConfig',
    'n_step_tboard_log '
    'n_step_console_log '
    'n_step_validation '
    'n_step_save_params '
    'n_step_visualize'
)


class GNodeTrainer:
    def __init__(self, gnode, dataloader, hyperparams, train_config, tensorboard_msg=''):
        # type: (GNode, BaseDataLoader, base_hp.Hyperparams, TrainConfig, str) -> None

        self.H = hyperparams
        self.train_config = train_config
        self.iter_no = 1
        self.dataloader = dataloader
        self.gnode = gnode

        # self.n_iter_gen = 0
        # self.n_iter_disc = 0
        # self.n_step_gen, self.n_step_disc = self.H.step_ratio
        # self.train_generator = True

        self.tensorboard_msg = tensorboard_msg

        # self.relabel_data()

    def relabel_data(self):
        train_data = self.dataloader.data['train']
        test_data = self.dataloader.data['test']
        train_labels = self.dataloader.labels['train']
        test_labels = self.dataloader.labels['test']

        train_splits, i_train_splits = self.gnode.split_x(train_data)
        test_splits, i_test_splits = self.gnode.split_x(test_data)

        print('train_splits:', map(lambda k: len(i_train_splits[k]), i_train_splits))
        print('test_splits:', map(lambda k: len(i_test_splits[k]), i_test_splits))

        for child_id in self.gnode.child_ids:
            data = (train_splits[child_id],
                    test_splits[child_id],
                    train_labels[i_train_splits[child_id]],
                    test_labels[i_test_splits[child_id]])

            dl = CustomDataLoader.create_from_parent(self.dataloader, data)
            dl.shuffle('train')

            cnode = self.gnode.child_nodes[child_id]  # type: GNode

            if cnode.trainer is None:
                cnode.set_trainer(dl, self.H, self.train_config, self.tensorboard_msg)
            else:
                cnode.get_trainer().data_loader = dl

    def full_train_step(self, n_iters=100):
        for child_id in self.gnode.child_ids:
            node = self.gnode.child_nodes[child_id]
            node.train(n_iters, enable_tqdm=False)

    def resume(self, dir_name, label, iter_no, n_iterations=None):
        # TODO: implement this
        self.iter_no = iter_no
        self.model.load_params(dir_name, label, iter_no)
        self.train(n_iterations)

    def train(self, n_iterations):
        n_iterations = n_iterations
        start_iter = self.iter_no
        end_iter = start_iter + n_iterations + 1

        with tqdm(total=n_iterations) as pbar:
            for self.iter_no in range(start_iter, end_iter):
                self.full_train_step()
                self.relabel_data()
                pbar.update(1)
