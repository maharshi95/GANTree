from __future__ import print_function, division
import os, argparse, logging, json
import traceback

from termcolor import colored

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys, time
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult

from tqdm import tqdm
import numpy as np

import torch as tr
from matplotlib import pyplot as plt
from configs import Config



default_args_str = '-hp hyperparams/digit_mnist_64.py -en mnist_mode_split -t'

if Config.use_gpu:
    print('mode: GPU')
    tr.set_default_tensor_type('torch.cuda.FloatTensor')

# Argument Parsing
parser = argparse.ArgumentParser()

parser.add_argument('-g', '--gpu', default=1, help='index of the gpu to be used. default: 0')
parser.add_argument('-t', '--tensorboard', default=False, const=True, nargs='?', help='Start Tensorboard with the experiment')
parser.add_argument('-r', '--resume', nargs='?', const=True, default=False,
                    help='if present, the training resumes from the latest step, '
                         'for custom step number, provide it as argument value')
parser.add_argument('-d', '--delete', nargs='+', default=[], choices=['logs', 'weights', 'results', 'all'],
                    help='delete the entities')
parser.add_argument('-w', '--weights', nargs='?', default='iter', choices=['iter', 'best_gen', 'best_pred'],
                    help='weight type to load if resume flag is provided. default: iter')
parser.add_argument('-hp', '--hyperparams', required=True, help='hyperparam class to use from HyperparamFactory')
parser.add_argument('-en', '--exp_name', default=None, help='experiment name. if not provided, it is taken from Hyperparams')

args = parser.parse_args(default_args_str.split()) if len(sys.argv) == 1 else parser.parse_args()

print(json.dumps(args.__dict__, indent=2))

resume_flag = args.resume is not False

from exp_context import ExperimentContext

###### Set Experiment Context ######
ExperimentContext.set_context(args.hyperparams, args.exp_name)
H = ExperimentContext.Hyperparams  # type: Hyperparams
exp_name = ExperimentContext.exp_name

from dataloaders.custom_loader import CustomDataLoader
from utils.tr_utils import as_np
from utils.viz_utils import get_x_clf_figure

##########  Set Logging  ###########
logger = logging.getLogger(__name__)
LOG_FORMAT = "[{}: %(filename)s: %(lineno)3s] %(levelname)s: %(funcName)s(): %(message)s".format(ExperimentContext.exp_name)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

gpu_idx = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

#### Clear Logs and Results based on the argument flags ####
from paths import Paths
from utils import bash_utils, model_utils

if 'all' in args.delete or 'logs' in args.delete or resume_flag is False:
    logger.warning('Deleting Logs...')
    bash_utils.delete_recursive(Paths.logs_base_dir)
    print('')

if 'all' in args.delete or 'results' in args.delete:
    logger.warning('Deleting all results in {}...'.format(Paths.results_base_dir))
    bash_utils.delete_recursive(Paths.results_base_dir)
    print('')

##### Create required directories
model_utils.setup_dirs()

##### Model and Training related imports
from dataloaders.factory import DataLoaderFactory
from base.hyperparams import Hyperparams
from trainers.gan_image_trainer import GanImgTrainer
from models.fashion.gan import ImgGAN
from models.toy.gt.gantree import GanTree
from models.toy.gt.gnode import GNode
from models.toy.gt.utils import DistParams

from trainers.gan_trainer import TrainConfig

# Dump Hyperparams file the experiments directory
hyperparams_string_content = json.dumps(H.__dict__, default=lambda x: repr(x), indent=4, sort_keys=True)
# print(hyperparams_string_content)
with open(Paths.exp_hyperparams_file, "w") as fp:
    fp.write(hyperparams_string_content)

train_config = TrainConfig(
    n_step_tboard_log=10,
    n_step_console_log=-1,
    n_step_validation=100,
    n_step_save_params=2000,
    n_step_visualize=1000
)


def get_data_tuple(data_dict, labels_dict=None):
    train_splits, train_split_index = root.split_x(data_dict['train'])
    test_splits, test_split_index = root.split_x(data_dict['test'])

    index = 4 if labels_dict else 2

    data_tuples = {
        i: (
               train_splits[i],
               test_splits[i],
               labels_dict['train'][train_split_index[i]] if labels_dict else None,
               labels_dict['test'][test_split_index[i]] if labels_dict else None
           )[:index] for i in root.child_ids
    }
    return data_tuples


def relabel_samples(node):
    dl = dl_set[node.id]
    full_data_tuples = get_data_tuple(dl.data, dl.labels)

    for i in node.child_ids:
        dl_set[i] = CustomDataLoader.create_from_parent(dl, full_data_tuples[i])


def split_dataloader(node):
    dl = dl_set[node.id]

    train_splits, train_split_index = node.split_x(dl.data['train'].cuda(), Z_flag=True)
    test_splits, test_split_index = node.split_x(dl.data['test'].cuda(), Z_flag=True)

    index = 4 if dl.supervised else 2

    for i in node.child_ids:
        train_labels = dl.labels['train'][train_split_index[i]] if dl.supervised else None
        test_labels = dl.labels['test'][test_split_index[i]] if dl.supervised else None

        data_tuples = (
            train_splits[i],
            test_splits[i],
            train_labels,
            test_labels
        )

        print(train_splits[i].shape)
        print(test_splits[i].shape)
        print(train_labels.shape)
        print(test_labels.shape)

        dl_set[i] = CustomDataLoader.create_from_parent(dl, data_tuples[:index])


def get_x_clf_plot_data(root, x_batch):
    with tr.no_grad():
        z_batch_post = tr.from_numpy(root.post_gmm_encode(x_batch)).cuda()
        x_recon_post, _ = root.post_gmm_decode(z_batch_post, train = False)

        z_batch_pre = tr.from_numpy(root.pre_gmm_encode(x_batch)).cuda()
        x_recon_pre = root.pre_gmm_decode(z_batch_pre)

        z_batch_post = as_np(z_batch_post)
        x_recon_post = as_np(x_recon_post)

        z_batch_pre = as_np(z_batch_pre)
        x_recon_pre = as_np(x_recon_pre)

    return z_batch_pre, z_batch_post, x_recon_pre, x_recon_post

#  node 0
def get_recons_data(node, x_batch):
    # type: (int, GNode, np.ndarray, np.ndarray) -> list
    z_batch_pre, z_batch_post, x_recon_pre, x_recon_post = get_x_clf_plot_data(node, x_batch)
    pred_post = node.gmm_predict(as_np(z_batch_post))

    print(x_recon_post.shape)
    

gan = ImgGAN.create_from_hyperparams('node0', H, '0')
means = as_np(gan.z_op_params.means)
cov = as_np(gan.z_op_params.cov)
dist_params = DistParams(means=means, cov=cov, pi=1.0, prob=1.0)

dl = DataLoaderFactory.get_dataloader(H.dataloader, H.batch_size, H.batch_size)

x_seed, l_seed = dl.random_batch('test', 512)

tree = GanTree('gtree', ImgGAN, H, x_seed)
root = tree.create_child_node(dist_params, gan)

root.set_trainer(dl, H, train_config, Model=GanImgTrainer)

# GNode.load('../experiments/mxp_1_trial/best_node-10.pt', root)
# GNode.load('best_node.pickle', root)
# for i in range(20):
#     root.train(5000)
#     root.save('../experiments/' + exp_name + '/best_node-' + str(i) + '.pt')

dl_set = {0: dl}
leaf_nodes = {0}
future_objects = []  # type: list[ApplyResult]
#
bash_utils.create_dir(Paths.weight_dir_path(''), log_flag=False)
pool = Pool(processes=16)

GNode.load('best_root_phase1_mnistdc.pickle', root)

get_recons_data(root, x_seed)