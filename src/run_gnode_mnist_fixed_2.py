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

import random

default_args_str = '-hp hyperparams/digit_mnist_64.py -en mnist_mode_split_fixed_2 -t'

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

##### Tensorboard Port
if args.tensorboard:
    ip = bash_utils.get_ip_address()
    tboard_port = str(bash_utils.find_free_port(Config.base_port))
    bash_utils.launchTensorBoard(Paths.logs_base_dir, tboard_port)
    address = '{ip}:{port}'.format(ip=ip, port=tboard_port)
    address_str = 'http://{}'.format(address)
    tensorboard_msg = "Tensorboard active at http://%s:%s" % (ip, tboard_port)
    html_content = """
    <h5>
        <b>Tensorboard hosted at 
            <a href={}>{}</a>
        </b>
    </h5>
    """.format(address_str, address)
    from IPython.core.display import display, HTML

    display(HTML(html_content))

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


def full_train_step(gnode, dl, visualize=True, validation=True, save_params=True):
    """
    :type gnode: GNode
    """
    trainer = gnode.trainer
    model = gnode.gan  # type: ImgGAN
    H = trainer.H

    trainer.iter_no += 1

    iter_time_start = time.time()

    x_train, _ = dl.next_batch('train')
    z_train = gnode.sample_z_batch(x_train.shape[0])

    x_train_ae = x_train
    z_train_ae = z_train

    trainer.train_step_ae(x_train_ae, z_train_ae)
    trainer.train_step_ad(x_train, z_train)

    # Train Losses Computation
    metrics = model.compute_metrics(x_train, z_train)
    g_acc, d_acc = metrics['accuracy_gen_x'], metrics['accuracy_dis_x']

    # Console Log
    if trainer.is_console_log_step():
        print('============================================================')
        print('Train Step', trainer.iter_no + 1)
        print('%s: step %i:     Disc Acc: %.3f' % (exp_name, trainer.iter_no, metrics['accuracy_dis_x'].item()))
        print('%s: step %i:     Gen  Acc: %.3f' % (exp_name, trainer.iter_no, metrics['accuracy_gen_x'].item()))
        print('%s: step %i: x_recon Loss: %.3f' % (exp_name, trainer.iter_no, metrics['loss_x_recon'].item()))
        print('%s: step %i: z_recon Loss: %.3f' % (exp_name, trainer.iter_no, metrics['loss_z_recon'].item()))
        print('------------------------------------------------------------')

    # Tensorboard Log
    if trainer.is_tboard_log_step():
        for tag, value in metrics.items():
            trainer.writer['train'].add_scalar(tag, value.item(), trainer.iter_no)

    # Validation Computations
    if validation and trainer.is_validation_step():
        trainer.validation()

    # Weights Saving
    if save_params and trainer.is_params_save_step():
        tic_save = time.time()
        model.save_params(dir_name='iter', weight_label='iter', iter_no=trainer.iter_no)
        tac_save = time.time()
        save_time = tac_save - tic_save
        if trainer.is_console_log_step():
            print('Param Save Time: %.4f' % (save_time))
            print('------------------------------------------------------------')

    # Visualization
    if visualize and trainer.is_visualization_step():
        # previous_backend = plt.get_backend()
        # plt.switch_backend('Agg')
        trainer.visualize('train')
        trainer.visualize('test')
        # plt.switch_backend(previous_backend)

    # Switch Training Networks - Gen | Disc
    trainer.switch_train_mode(g_acc, d_acc)

    iter_time_end = time.time()
    if trainer.is_console_log_step():
        print('Total Iter Time: %.4f' % (iter_time_end - iter_time_start))
        if trainer.tensorboard_msg:
            print('------------------------------------------------------------')
            print(trainer.tensorboard_msg)
        print('============================================================')
        print()


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
        # seed_data[i] = dl_set[i].random_batch('train', 64)
    # seed_data_pkfile = 'seed_data-13.pickle'
    # with open(seed_data_pkfile, 'w') as fp:
    #     pickle.dump(seed_data, fp)


# nodes[1].update_dist_params(means=2 * np.ones(2), cov=np.eye(2), prior_prob=1)
# nodes[2].update_dist_params(means=- 2 * np.ones(2), cov=np.eye(2), prior_prob=1)

def get_x_clf_plot_data(root, x_batch):
    with tr.no_grad():
        z_batch_post = tr.from_numpy(root.post_gmm_encode(x_batch)).cuda()
        x_recon_post, _ = root.post_gmm_decode(z_batch_post, train = False)

        z_batch_pre = tr.from_numpy(root.pre_gmm_encode(x_batch)).cuda()
        x_recon_pre = root.pre_gmm_decode(z_batch_pre)

        z_batch_post = as_np(z_batch_post.cpu())
        x_recon_post = as_np(x_recon_post)

        z_batch_pre = as_np(z_batch_pre.cpu())
        x_recon_pre = as_np(x_recon_pre)

    return z_batch_pre, z_batch_post, x_recon_pre, x_recon_post


def get_deep_type(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [get_deep_type(o) for o in obj]
    return str(type(obj))

from sklearn.decomposition import PCA
from torch import distributions as dist
from torch.nn import functional as F


def get_plot_data(iter_no, node, x_batch, labels, threshold, pca2rand = None, pca = False):
    # type: (int, GNode, np.ndarray, np.ndarray) -> list
    z_batch_pre, z_batch_post, x_recon_pre, x_recon_post = get_x_clf_plot_data(node, x_batch)
    pred_post = node.gmm_predict_test(z_batch_post, threshold = threshold)

    l_seed_ch0 = labels[np.where(pred_post == 0)]
    l_seed_ch1 = labels[np.where(pred_post == 1)]

    count_ch0 = [0 for i in range(10)]
    count_ch1 = [0 for i in range(10)]
    prob_ch0 = [0 for i in range(10)]
    prob_ch1 = [0 for i in range(10)]

    for i in l_seed_ch0:
        count_ch0[i] += 1

    for i in l_seed_ch1:
        count_ch1[i] += 1

    for i in range(10):
        if ((count_ch0[i] + count_ch1[i]) != 0.0):
            prob_ch0[i] = count_ch0[i] * 1.0 / (count_ch0[i] + count_ch1[i])
            prob_ch1[i] = count_ch1[i] * 1.0 / (count_ch0[i] + count_ch1[i])
        else:
            prob_ch0[i] = 0
            prob_ch1[i] = 0


    z_rand0 = node.get_child(0).sample_z_batch(x_batch.shape[0])
    z_rand1 = node.get_child(1).sample_z_batch(x_batch.shape[0])
    with tr.no_grad():
        x_fake0 = node.post_gmm_decoders[0].forward(z_rand0)
        x_fake1 = node.post_gmm_decoders[1].forward(z_rand1)

    if pca:
        # pca2 = PCA(n_components = 2)
        # if pca2rand == None:
            # print("None")
        pca2rand = PCA(n_components = 2)
        
        # pca2rand.fit(np.append(np.append(z_rand1, z_rand0, axis = 0), z_batch_post, axis = 0))

        # pca2rand.fit(z_batch_post)
        pca2rand.fit(np.append(z_rand1, z_rand0, axis = 0))

            # print("Same")

        # pca2.fit(np.append(z_batch_pre, z_batch_post, axis = 0))
        covariance_trans = np.matmul(pca2rand.components_, np.cov(np.transpose(z_batch_post)))
        # covariance_trans = np.matmul(pca2rand.components_, np.cov(np.transpose(np.append(z_rand1, z_rand0, axis = 0))))
        covariance_trans = np.matmul(covariance_trans, np.transpose(pca2rand.components_))

        child_0_cov = np.matmul(pca2rand.components_, node.kmeans.covs[0])
        child_0_cov = np.matmul(child_0_cov, np.transpose(pca2rand.components_))
        child_1_cov = np.matmul(pca2rand.components_, node.kmeans.covs[1])
        child_1_cov = np.matmul(child_1_cov, np.transpose(pca2rand.components_))

        print(covariance_trans)
        print(node.get_child(0).prior_means)
        print(node.get_child(1).prior_means)
        # print(node.get_child(0).prior_means.reshape(1, -1))
        # print(node.get_child(1).prior_means.reshape(1, -1))
        print(pca2rand.transform(node.get_child(0).prior_means.reshape(1, -1)))
        print(pca2rand.transform(node.get_child(1).prior_means.reshape(1, -1)))
        print(pca2rand.transform(node.get_child(0).prior_means.reshape(1, -1)).shape)
        print(pca2rand.transform(node.get_child(1).prior_means.reshape(1, -1)).shape)

        z_batch_pre_red = pca2rand.transform(z_batch_pre)
        z_batch_post_red = pca2rand.transform(z_batch_post)
        z_rand0_red = pca2rand.transform(z_rand0)
        z_rand1_red = pca2rand.transform(z_rand1)
        child_0_mean = pca2rand.transform(node.get_child(0).prior_means.reshape(1, -1))[0]
        child_1_mean = pca2rand.transform(node.get_child(1).prior_means.reshape(1, -1))[0]

        # z_batch_pre_red = np.matmul(pca2rand.transform(z_batch_pre), np.transpose(covariance_trans))
        # z_batch_post_red = np.matmul(pca2rand.transform(z_batch_post), np.transpose(covariance_trans))
        # z_rand0_red = np.matmul(pca2rand.transform(z_rand0), np.transpose(covariance_trans))
        # z_rand1_red = np.matmul(pca2rand.transform(z_rand1), np.transpose(covariance_trans))
        # child_0_mean = np.matmul(pca2rand.transform(node.get_child(0).prior_means.reshape(1, -1))[0], np.transpose(covariance_trans))
        # child_1_mean = np.matmul(pca2rand.transform(node.get_child(1).prior_means.reshape(1, -1))[0], np.transpose(covariance_trans))

        # print(child_0_mean)
        # print(child_1_mean)

        print(distance.mahalanobis(child_0_mean, child_1_mean, child_1_cov))
        print(distance.mahalanobis(child_1_mean, child_0_mean, child_0_cov))

    else:
        z_batch_pre_red = z_batch_pre[:, 0:2]
        z_batch_post_red = z_batch_post[:, 0:2]
        z_rand0_red = z_rand0[:, 0:2]
        z_rand1_red = z_rand1[:, 0:2] 
        child_0_mean = node.get_child(0).prior_means[0:2]
        child_1_mean = node.get_child(1).prior_means[0:2]

    plot_data = [
        [
            [
                as_np(x_batch),
                as_np(pred_post).astype(int)
            ],
            as_np(node.dist_params),
            node.get_child(0).dist_params,
            node.get_child(1).dist_params
        ], [
            z_batch_pre_red,
            z_batch_post_red,
            x_recon_pre,
            x_recon_post
        ], [
            as_np(z_rand0_red),
            as_np(x_fake0),
            as_np(z_rand1_red),
            as_np(x_fake1),
        ], [
            as_np(child_0_mean),
            as_np(child_1_mean),
            as_np(child_0_cov),
            as_np(child_1_cov)
        ], [
            as_np(prob_ch0),
            as_np(prob_ch1)
        ], (threshold / np.sqrt(z_batch_post.shape[0]))
    ]
    return plot_data, pca2rand


def generate_plots(plot_data, iter_no, tag, node):
    fig, fig1 = get_x_clf_figure(plot_data, n_modes=10)

    node.trainer.writer['test'].add_figure('plots', fig, iter_no)
    node.trainer.writer['test'].add_figure('confidence', fig1, iter_no)

    path = Paths.get_result_path('test_plots/%s_%03d' % ('test_plot', iter_no))
    fig.savefig(path)
    path1 = Paths.get_result_path('test_confidence/%s_%03d' %('test_confidence', iter_no))
    fig1.savefig(path1)
    # plt.close(fig)
    # plt.close(fig1)
    return fig, fig1


# def visualize_plots(iter_no, node, x_batch, labels, tag, pca2rand = None):
#     plot_data, pca2rand = get_plot_data(iter_no, node, x_batch, labels, pca2rand = pca2rand, pca = True)
#     future = pool.apply_async(generate_plots, (plot_data, iter_no, tag, node))
#     generate_plots(plot_data, iter_no, tag, node)
#     future_objects.append(future)
#     return future, pca2rand

def visualize_plots(iter_no, node, x_batch, labels, tag, threshold, pca2rand = None):
    plot_data, pca2rand = get_plot_data(iter_no, node, x_batch, labels, threshold, pca2rand = pca2rand, pca = True)
    generate_plots(plot_data, iter_no, tag, node)
    return pca2rand


def save_node(node, tag=None, iter=None):
    # type: (GNode, str, int) -> None
    filename = node.name
    if tag is not None:
        filename += '_' + str(tag)
    if iter is not None:
        filename += ('_%05d' % iter)
    filename = filename + '.pt'
    filepath = os.path.join(Paths.weight_dir_path(''), filename)
    node.save(filepath)


def load_node(node_name, tag=None, iter=None):
    filename = node_name
    if tag is not None:
        filename += '_' + str(tag)
    if iter is not None:
        filename += ('_%05d' % iter)
    filepath = os.path.join(Paths.weight_dir_path(''), filename)
    gnode = GNode.load(filepath, Model=ImgGAN)
    return gnode


# batch_size multiple of 256
def get_z(node, batch_size):
    Z = tr.tensor([])

    iter = batch_size // 256

    for i in range(iter):
        x = dl.random_batch(split='train', batch_size=256)[0]
        z = tr.from_numpy(node.post_gmm_encode(x)).cuda()
        Z = tr.cat((Z, z), 0)
        # print('gmm iter',iter)
    return Z

def get_data(node, split):
    Z = tr.tensor([])

    if split == 'train':
        data_full = dl.train_data()
        data = data_full[np.where(dl.train_data_labels() != 1)]

    # print(len(data))

    iter = (len(data) // 256) + 1

    for i in range(iter):
        if i < iter -1:
            x = data[i*256:(i+1)*256]
        else:
            x = data[i*256:]
        z = tr.from_numpy(node.post_gmm_encode(x)).cuda()
        Z = tr.cat((Z,z), 0)


    # print(len(Z))
    return Z

# epoch-wise
from scipy.spatial import distance

def train_phase_1(node, epochs):
    # print('entered phase 1')

    threshold = 3.0
    phase1_epochs = 7
    percentile = 0.08
    limit = 100
    # train_data = dl_set[0].train_data()
    # train_data_labels = dl_set[0].train_data_labels()

    train_data_full = dl_set[0].train_data()
    train_data_labels_full = dl_set[0].train_data_labels()

    train_data = train_data_full[np.where(train_data_labels_full != 1)]
    train_data_labels = train_data_labels_full[np.where(train_data_labels_full != 1)]


    eigenvalues = node.kmeans.pca.explained_variance_

    total_eigenvalues = sum(eigenvalues)

    summed = [sum(eigenvalues[:i]) for i in range(1, len(eigenvalues)+1)]

    # print(summed)
    # print(len(summed))

    summed = summed / total_eigenvalues

    # print(summed)

    plt.plot(summed, 'ro')
    fig_eigen = plt.gcf()
    node.trainer.writer['train'].add_figure('eigenvalues', fig_eigen, 0)
    path_eigen = Paths.get_result_path('eigenvalues/%s_%03d' % ('eigenvalues', 0))
    fig_eigen.savefig(path_eigen)

    pca2rand = visualize_plots(iter_no=0, node=node, x_batch=x_seed, labels=l_seed, tag='x_clf_plots', threshold = threshold)

    baseline_vector = tr.Tensor([0.0 for i in range(100)])
    baseline_vector[0] = threshold
    f1 = dist.MultivariateNormal(tr.Tensor([0.0 for i in range(100)]), tr.Tensor(node.kmeans.covs[0]))
    baseline = -f1.log_prob(baseline_vector)

    training_list = []
    assigned_index = []
    unassigned_index = []

    # node = GNode.loadwithkmeans('best_root_phase1_mnistdc_'+ str(phase1_epochs) +'.pickle', node)
    # child0 = GNode.loadwithkmeans('best_child0_phase1_mnistdc_'+ str(phase1_epochs) + '.pickle', node.get_child(0))
    # child1 = GNode.loadwithkmeans('best_child1_phase1_mnistdc_'+ str(phase1_epochs) + '.pickle', node.get_child(1))

    # node.child_nodes = {1: child0, 2: child1}


    for j in range(phase1_epochs, epochs):
    # for j in range(0, epochs):
        print("epoch number: " + str(j))
        Z = get_data(node = node, split = 'train')
        batchSize = dl_set[0].batch_size['train']
        if len(Z) % batchSize == 0:
            n_iterations = (len(Z) // batchSize) 
        else:
            n_iterations = (len(Z) // batchSize) + 1

        # node.save('best_root_phase1_mnistdc_'+ str(j) +'.pickle')
        # node.get_child(0).save('best_child0_phase1_mnistdc_'+ str(j) + '.pickle')
        # node.get_child(1).save('best_child1_phase1_mnistdc_' + str(j) + '.pickle')

        

        # print(batchSize)
        # print(n_iterations)
        # simcrossdist = node.update_child_params(x_seed, Z= Z.cpu().numpy(), max_iter=5)
        
        # node.trainer.writer['train'].add_scalar('simcrossdist', simcrossdist, j * n_iterations)
        # mean_time_taken = 1.0

        if j == phase1_epochs:
            node.reassignLabels(train_data, threshold, reassignLabels = True)
            # node.assignLabels(X = train_data, percentile = percentile, limit = limit, reassignLabels = True)


        with tqdm(total=n_iterations) as pbar:
            for iter_no in range(n_iterations):
                tic = time.time()
                current_iter_no = j * n_iterations + iter_no
                node.trainer.iter_no = current_iter_no

                if (current_iter_no % 200 == 0) and (j >= phase1_epochs):

                    node.assignLabels(X = train_data, percentile = percentile, limit = limit, reassignLabels = False)

                    # p = node.kmeans.pred
                    # cluster_distances = []
                    # cluster_labels = []

                    # for i in range(len(p)):
                    #     if p[i] == 2:
                    #         dis0 = distance.mahalanobis(Z.detach().cpu().numpy()[i], node.kmeans.means[0], node.kmeans.covs[0])
                    #         dis1 = distance.mahalanobis(Z.detach().cpu().numpy()[i], node.kmeans.means[1], node.kmeans.covs[1])

                    #         d = min(dis0/dis1, dis1/dis0)
                    #         cluster_distances.append(d)

                    #         if d == dis0/dis1:
                    #             cluster_labels.append(0)
                    #         elif d == dis1/dis0:
                    #             cluster_labels.append(1)

                    # sorted_list = np.argsort(cluster_distances)

                    # total_influx = percentile * len(sorted_list)

                    # if total_influx < limit:
                    #     total_influx = min(limit, len(sorted_list))
                    #     update_list = sorted_list[:total_influx]
                    # else:
                    #     update_list = sorted_list[:total_influx]

                    # p[update_list] = cluster_labels[update_list]

                    # node.kmeans.pred = p
                #         if p[i] == 2:
                #             if (dis0 < dis1) and (dis0 < threshold):
                #                 p[i] = 0
                #             elif (dis0 > dis1) and (dis1 < threshold):
                #                 p[i] = 1

                #     node.kmeans.pred = p

                # Training common encoder over cross-classification loss with a batch across common dataloader
                if (j >= 0) and (current_iter_no % 200 == 0):
                    # free some labels
                    # if j == 5 and current_iter_no == 4400:
                    # if j == 0 and current_iter_no == 400:
                        # node.reassignLabels(train_data, threshold)

                    p = node.kmeans.pred

                    unassigned_labels = [0 for i in range(10)]
                    assigned_labels = [0 for i in range(10)]

                    for i in range(len(p)):
                        if p[i] == 2:
                            unassigned_labels[train_data_labels[i]] += 1
                        else:
                            assigned_labels[train_data_labels[i]] += 1

                    barWidth = 0.3
                    r1 = np.arange(len(unassigned_labels))
                    r2 = [x+barWidth for x in r1]

                    plt.bar(r1, unassigned_labels, width = barWidth, color = 'purple', edgecolor = 'black', capsize=7)
                 
                    plt.bar(r2, assigned_labels, width = barWidth, color = 'green', edgecolor = 'black', capsize=7)
                 
                    # general layout
                    plt.xticks([r + barWidth for r in range(len(unassigned_labels))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                    plt.ylabel('count')

                    fig_train_assigned = plt.gcf()
                    node.trainer.writer['train'].add_figure('assigned_labels_count', fig_train_assigned, current_iter_no)
                    path_assign_train = Paths.get_result_path('train_assigned/%s_%03d' % ('train_assigned', current_iter_no))
                    fig_train_assigned.savefig(path_assign_train)

                    l_seed_ch0 = train_data_labels[np.where(p == 0)]
                    l_seed_ch1 = train_data_labels[np.where(p == 1)]

                    count_ch0 = [0 for i in range(10)]
                    count_ch1 = [0 for i in range(10)]
                    prob_ch0 = [0 for i in range(10)]
                    prob_ch1 = [0 for i in range(10)]

                    for i in l_seed_ch0:
                        count_ch0[i] += 1

                    for i in l_seed_ch1:
                        count_ch1[i] += 1

                    for i in range(10):
                        if (count_ch0[i] + count_ch1[i]) != 0:
                            prob_ch0[i] = count_ch0[i] * 1.0 / (count_ch0[i] + count_ch1[i])
                            prob_ch1[i] = count_ch1[i] * 1.0 / (count_ch0[i] + count_ch1[i])
                        else:
                            prob_ch0[i] = 0
                            prob_ch1[i] = 0

                    plt.bar(r1, prob_ch0, width = barWidth, color = 'red', edgecolor = 'black', capsize=7)
                    plt.bar(r2, prob_ch1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7)
                    plt.xticks([r + barWidth for r in range(len(prob_ch0))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                    plt.ylabel('count')

                    fig_confidence_train = plt.gcf()
                    node.trainer.writer['train'].add_figure('confidence', fig_confidence_train, current_iter_no)
                    path_confidence_train = Paths.get_result_path('train_confidence/%s_%03d' % ('train_confidence', current_iter_no))
                    fig_confidence_train.savefig(path_confidence_train)

                    aboveThresholdLabels_ch0 = [0 for i in range(10)]
                    aboveThresholdLabels_ch1 = [0 for i in range(10)]

                    percentAbove_ch0 = [0 for i in range(10)]
                    percentAbove_ch1 = [0 for i in range(10)]

                    for i in range(len(p)):
                        if p[i] == 0:
                            if (distance.mahalanobis(Z[i], node.kmeans.means[0], node.kmeans.covs[0])) > threshold:
                                aboveThresholdLabels_ch0[train_data_labels[i]] += 1
                        elif p[i] == 1:
                            if (distance.mahalanobis(Z[i], node.kmeans.means[1], node.kmeans.covs[1])) > threshold:
                                aboveThresholdLabels_ch1[train_data_labels[i]] += 1

                    for i in range(10):
                        if (count_ch0[i]) != 0:
                            percentAbove_ch0[i] = aboveThresholdLabels_ch0[i] * 1.0 / count_ch0[i]
                        else:
                            percentAbove_ch0[i] = 0

                        if (count_ch1[i] != 0):
                            percentAbove_ch1[i] = aboveThresholdLabels_ch1[i] * 1.0 / count_ch1[i]
                        else:
                            percentAbove_ch1[i] = 0

                    plt.bar(r1, aboveThresholdLabels_ch0, width = barWidth, color = 'red', edgecolor = 'black', capsize=7)
                    plt.bar(r2, aboveThresholdLabels_ch1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7)
                    plt.xticks([r + barWidth for r in range(len(percentAbove_ch0))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                    plt.ylabel('count')

                    fig_above_threshold_train = plt.gcf()
                    node.trainer.writer['train'].add_figure('aboveThresholdTrain', fig_above_threshold_train, current_iter_no)
                    path_above_threshold_train = Paths.get_result_path('train_above_threshold/%s_%03d' % ('above_threshold_train', current_iter_no))
                    fig_above_threshold_train.savefig(path_above_threshold_train)






                if (iter_no < (n_iterations - 1)) and j < phase1_epochs:
                    print("in")
                    start_no = iter_no * batchSize
                    end_no = start_no + batchSize
                    training_list = [i for i in range(start_no, end_no)]
                    # x_clf_train_batch = train_data[start_no:end_no]
                    x_clf_train_batch = train_data[training_list]
                elif j < phase1_epochs:
                    print("out")
                    start_no = iter_no * batchSize
                    end_no = len(train_data)
                    training_list = [i for i in range(start_no, end_no)]
                    # x_clf_train_batch = train_data[start_no:end_no]
                    x_clf_train_batch = train_data[training_list]
                else:
                    print('cccccc')
                    p = node.kmeans.pred
                    # print(p)

                    # print(np.where(p != 2))
                    # print(np.where(p == 2))

                    assigned_index = np.where(p != 2)[0]
                    length_assi = len(assigned_index)
                    # print(assigned_index)

                    unassigned_index = np.where(p == 2)[0]
                    length_unassi = len(unassigned_index)
                    # print(unassigned_index)

                    # print(length_assi)
                    # print(length_unassi)

                    if length_assi == 0 or length_unassi == 0:
                        start_no = iter_no * batchSize
                        end_no = start_no + batchSize
                        training_list = [i for i in range(start_no, end_no)]
                    elif batchSize/2 > length_assi or batchSize/2 > length_unassi:
                        training_list = random.sample(assigned_index, min(length_unassi, length_assi))
                        training_list.extend(random.sample(unassigned_index, min(length_assi, length_unassi)))
                    else:

                        start_no_assi = int((iter_no * batchSize / 2) % length_assi)
                        end_no_assi = int((start_no_assi + batchSize/2) % length_assi)

                        # print(start_no_assi)
                        # print(end_no_assi)

                        if end_no_assi < start_no_assi:
                            training_list = [i for i in assigned_index[start_no_assi:]]
                            training_list.extend([i for i in assigned_index[:end_no_assi]])
                        else:
                            training_list = [i for i in assigned_index[start_no_assi:end_no_assi]]

                        # print(assigned_index[start_no_assi:end_no_assi])

                        start_no_unassi = int((iter_no * batchSize/2) % length_unassi)
                        end_no_unassi = int((start_no_unassi + batchSize/2) % length_unassi)

                        # print(start_no_unassi)
                        # print(end_no_unassi)

                        if end_no_unassi < start_no_unassi:
                            training_list.extend([i for i in unassigned_index[start_no_unassi:]])
                            training_list.extend([i for i in unassigned_index[:end_no_unassi]])
                        else:
                            training_list.extend([i for i in unassigned_index[start_no_unassi:end_no_unassi]])

                        # print(unassigned_index[start_no_unassi:end_no_unassi])
                        # training_list = random.sample(np.where(p != 2), batchSize/2)
                        # training_list.append(random.sample(np.where(p == 2)), batchSize/2)
                    # print(training_list)
                    x_clf_train_batch = train_data[training_list]

                # print(training_list)

                # print(x_clf_train_batch.shape)

                z_batch, x_recon, preds, x_clf_loss_assigned, x_assigned_recon_loss, loss_assigned, x_clf_loss_unassigned, x_unassigned_recon_loss, loss_unassigned, x_clf_cross_loss, loss_recon, loss_log_assigned_ch0, loss_log_assigned_ch1 = node.step_train_x_clf(x_clf_train_batch, training_list, w1 = 1.0, w2 = 1.0, w3 = 0.0, w4 = 0.0, with_PCA = True, threshold = threshold)

                # if (current_iter_no % 300 == 0):
                #     print(distance.mahalanobis(z_batch.detach().cpu().numpy()[3], node.kmeans.means[0], node.kmeans.covs[0]))

                positive = np.sum(preds == 0.)
                negative = np.sum(preds == 1.)

                neutral = np.sum(preds == 2.)

                # ratio = positive * 1.0 / (max((len(preds) - neutral), 1e-9))
                ratio = positive * 1.0 / (positive + negative)

                unassigned_ratio = neutral * 1.0 / len(preds)

                if current_iter_no % 10 == 0:
                    node.trainer.writer['train'].add_scalar('x_clf_loss_assigned', x_clf_loss_assigned, current_iter_no)
                    node.trainer.writer['train'].add_scalar('x_assigned_recon_loss', x_assigned_recon_loss, current_iter_no)
                    node.trainer.writer['train'].add_scalar('loss_assigned', loss_assigned, current_iter_no)
                    node.trainer.writer['train'].add_scalar('split_ratio', ratio, current_iter_no)
                    node.trainer.writer['train'].add_scalar('x_clf_cross_loss', x_clf_cross_loss, current_iter_no)
                    node.trainer.writer['train'].add_scalar('unassigned_ratio', unassigned_ratio, current_iter_no)
                    node.trainer.writer['train'].add_scalar('x_clf_loss_unassigned', x_clf_loss_unassigned, current_iter_no)
                    node.trainer.writer['train'].add_scalar('x_unassigned_recon_loss', x_unassigned_recon_loss, current_iter_no)
                    node.trainer.writer['train'].add_scalar('loss_unassigned', loss_unassigned, current_iter_no)
                    node.trainer.writer['train'].add_scalar('loss_recon', loss_recon, current_iter_no)
                    node.trainer.writer['train'].add_scalar('loss_log_assigned_ch0', loss_log_assigned_ch0, current_iter_no)
                    node.trainer.writer['train'].add_scalar('loss_log_assigned_ch1', loss_log_assigned_ch1, current_iter_no)
                    node.trainer.writer['base'].add_scalar('loss_log_assigned_ch0', baseline, current_iter_no)
                    node.trainer.writer['base'].add_scalar('loss_log_assigned_ch1', baseline, current_iter_no)
                    

                    # if current_iter_no % 200 == 0:
                    #     print(distance.mahalanobis(z_batch.detach().cpu().numpy()[0], node.kmeans.means[0], node.kmeans.covs[0]))
                    #     print(distance.mahalanobis(z_batch.detach().cpu().numpy()[0], node.kmeans.means[1], node.kmeans.covs[1]))
                        

                    #     print(distance.mahalanobis(node.kmeans.means[0], node.kmeans.means[1], node.kmeans.covs[1]))
                    #     print(distance.mahalanobis(node.kmeans.means[1], node.kmeans.means[0], node.kmeans.covs[0]))

                # validation
                if current_iter_no % 40 == 0:

                    preds_test, x_clf_loss_assigned_test, x_assigned_recon_loss_test, loss_assigned_test, x_clf_loss_unassigned_test, x_unassigned_recon_loss_test, loss_unassigned_test, x_clf_cross_loss_test, loss_recon_test, loss_log_assigned_ch0_test, loss_log_assigned_ch1_test = node.step_predict_test(x_seed, with_PCA = True, threshold = threshold)

                    positive_test = np.sum(preds_test == 0.)
                    negative_test = np.sum(preds_test == 1.)

                    neutral_test = np.sum(preds_test == 2.)

                    ratio_test = positive_test * 1.0 / (max((len(preds_test) - neutral_test), 1e-9))
                    ratio_test = positive_test * 1.0 / (positive_test + negative_test)

                    unassigned_ratio_test = neutral_test * 1.0 / len(preds_test)

                    node.trainer.writer['test'].add_scalar('x_clf_loss_assigned', x_clf_loss_assigned_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar('x_assigned_recon_loss', x_assigned_recon_loss_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar('loss_assigned', loss_assigned_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar('split_ratio', ratio_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar('x_clf_cross_loss', x_clf_cross_loss_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar('unassigned_ratio', unassigned_ratio_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar('x_clf_loss_unassigned', x_clf_loss_unassigned_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar('x_unassigned_recon_loss', x_unassigned_recon_loss_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar('loss_unassigned', loss_unassigned_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar('loss_recon', loss_recon_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar('loss_log_assigned_ch0_test', loss_log_assigned_ch0_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar('loss_log_assigned_ch1_test', loss_log_assigned_ch1_test, current_iter_no)

                    if current_iter_no % 200 == 0:
                        unassigned_labels_test = [0 for i in range(10)]
                        assigned_labels_test = [0 for i in range(10)]

                        for i in range(len(preds_test)):
                            if preds_test[i] == 2:
                                unassigned_labels_test[l_seed[i]] += 1
                            else:
                                assigned_labels_test[l_seed[i]] += 1

                        barWidth = 0.3
                        r1 = np.arange(len(unassigned_labels_test))
                        r2 = [x+barWidth for x in r1]

                        plt.bar(r1, unassigned_labels_test, width = barWidth, color = 'purple', edgecolor = 'black', capsize=7)
                     
                        plt.bar(r2, assigned_labels_test, width = barWidth, color = 'green', edgecolor = 'black', capsize=7)
                     
                        # general layout
                        plt.xticks([r + barWidth for r in range(len(unassigned_labels_test))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                        plt.ylabel('count')

                        fig_test_assigned = plt.gcf()
                        node.trainer.writer['test'].add_figure('assigned_labels_count', fig_test_assigned, current_iter_no)
                        path_assigned_test = Paths.get_result_path('test_assigned/%s_%03d' % ('test_assigned', current_iter_no))
                        fig_test_assigned.savefig(path_assigned_test)

                pbar.update(n=1)

                if current_iter_no % 600 == 0 and current_iter_no != 0:
                    visualize_plots(iter_no=current_iter_no, node=node, x_batch=x_seed, labels=l_seed, tag='x_clf_plots', pca2rand = pca2rand, threshold = threshold)



# def train_phase_1(node, n_iterations):
#     # print('entered phase 1')
#     Z = get_z(node=node, batch_size=2048)
#     # print('train phase 1: got Z')
#     # node.fit_gmm(x_seed, Z=Z, max_iter=10)

#     _, pca2rand = visualize_plots(iter_no=0, node=node, x_batch=x_seed, labels=l_seed, tag='x_clf_plots')

#     mean_time_taken = 1.0

#     with tqdm(total=n_iterations) as pbar:
#         for iter_no in range(n_iterations):
#             tic = time.time()
#             node.trainer.iter_no = iter_no

#             i = iter_no + 1

#             # Training common encoder over cross-classification loss with a batch across common dataloader
#             x_clf_train_batch, _ = dl_set[0].next_batch('train')
#             z_batch, x_recon, x_recon_loss, x_clf_loss, loss, preds, time_taken = node.step_train_x_clf(x_clf_train_batch)
#             # mean_time_taken = 0.8 * mean_time_taken + 0.2 * time_taken

#             positive = np.sum(preds == 0.)
#             negative = np.sum(preds == 1.)

#             ratio = positive * 1.0 / (positive + negative)

#             if i % 10 == 0:
#                 node.trainer.writer['train'].add_scalar('x_clf_loss', x_clf_loss, iter_no)
#                 node.trainer.writer['train'].add_scalar('x_recon_loss', x_recon_loss, iter_no)
#                 node.trainer.writer['train'].add_scalar('loss', loss, iter_no)
#                 node.trainer.writer['train'].add_scalar('split_ratio', ratio, iter_no)

#             child0mean = node.get_child(0).prior_means
#             child1mean = node.get_child(1).prior_means
#             meanDistance = np.linalg.norm(child0mean - child1mean)

#             node.trainer.writer['train'].add_scalar('meanDistance', meanDistance, iter_no)

#             if iter_no % 10 == 0:
#                 print(meanDistance)

#             if meanDistance < 15.0:
#                 if iter_no % 10 == 0:
#                     Z = get_z(node=node, batch_size=2048)
#                     simcrossdist = node.update_child_params(x_seed, Z= Z.cpu().numpy(), max_iter=5)
#                     node.trainer.writer['train'].add_scalar('simcrossdist', simcrossdist, iter_no)

                

#             # markerline0, stemlines0, baseline0 = plt.stem(child0mean[4:6])
#             # plt.setp(stemlines0, color='r', linewidth=2)
#             # plt.setp(markerline0, color='r')
#             # markerline1, stemlines1, baseline1 = plt.stem(child1mean[4:6])
#             # plt.setp(stemlines1, color='b', linewidth=2)
#             # plt.setp(markerline1, color='b')
#             # node.trainer.writer['train'].add_figure('means', plt.gcf(), iter_no)
#             # plt.savefig('../experiments/gnode_mnist_plots/mean'+str(iter_no)+'.png')


#             pbar.update(n=1)

#             # tac = time.time()
#             # time_taken = tac - tic
#             # mean_time_taken = 0.8 * mean_time_taken + 0.2 * time_taken
#             # if i % 100 == 0:
#             #     print(mean_time_taken)

#             if i < 10 or i % 10 == 0:
#                 visualize_plots(iter_no=i, node=node, x_batch=x_seed, labels=l_seed, tag='x_clf_plots', pca2rand = pca2rand)



def is_gan_vis_iter(i):
    return (i < 50
            or (i < 1000 and i % 20 == 0)
            or (i < 5000 and i % 100 == 0)
            or (i % 500 == 0))


def train_phase_2(node, n_iterations):
    # type: (GNode, int) -> None
    with tqdm(total=n_iterations) as pbar:
        for iter_no in range(n_iterations):
            for i in node.child_ids:
                full_train_step(node.child_nodes[i], dl_set[i], visualize=False)
                pbar.update(n=0.5)
            # root.fit_gmm(x_seed)
            # if is_gan_vis_iter(iter_no):
            #     visualize_plots(iter_no, root, x_seed, l_seed, tag='gan_plots')


def train_node(node, x_clf_iters=200, gan_iters=10000):
    # type: (GNode, int, int) -> None
    global future_objects

    train_data_full = dl_set[0].train_data()
    train_data_labels_full = dl_set[0].train_data_labels()

    train_data = train_data_full[np.where(train_data_labels_full != 1)]

    child_nodes = tree.load_children(node, phase1_epochs)
    # child_nodes = tree.split_node(node, x_batch = train_data, fixed=False)
    # get_recons_data(root, x_seed, l_seed)

    train_phase_1(node, x_clf_iters)

    # nodes = {node.id: node for node in child_nodes}  # type: dict[int, GNode]

    # split_dataloader(node)

    # nodes[1].set_trainer(dl_set[1], H, train_config, Model=GanImgTrainer)
    # nodes[2].set_trainer(dl_set[2], H, train_config, Model=GanImgTrainer)

    # # future_objects = []  # type: list[ApplyResult]

    # # train_phase_2(node, gan_iters)
    # # Logging the image saving operations status
    # for i, obj in enumerate(future_objects):
    #     iter_no, path = obj.get()
    #     if obj.successful():
    #         print('Saved figure for iter %3d @ %s' % (iter_no, path))
    #     else:
    #         print('Failed saving figure for iter %d' % iter_no)


def likelihood(node, dl):
    samples = dl.data['train'].shape[0]
    # print('count of samples',samples)
    X_complete = dl.data['train'].cuda()
    iter = samples // 256
    p = np.zeros([iter], dtype=np.float32)

    for idx in range(iter):
        p[idx] = node.mean_likelihood(X_complete[(idx) * 256:(idx + 1) * 256])
    # print (p)
    return np.mean(p)


def find_next_node():
    logger.info(colored('Leaf Nodes: %s' % str(list(leaf_nodes)), 'green', attrs=['bold']))
    likelihoods = {i: likelihood(tree.nodes[i], dl_set[i]) for i in leaf_nodes}
    n_samples = {i: dl_set[i].data['train'].shape[0] for i in leaf_nodes}
    pairs = [(node_id, n_samples[node_id], likelihoods[node_id]) for node_id in leaf_nodes]
    for pair in pairs:
        logger.info('Node: %2d N_Samples: %5d Likelihood %.03f' % (pair[0], pair[1], pair[2]))
    min_samples = min(n_samples)
    for leaf_id in leaf_nodes:
        if n_samples[leaf_id] > 3 * min_samples:
            return max(leaf_nodes, key=lambda i: n_samples[i])
    return min(leaf_nodes, key=lambda i: likelihoods[i])

def get_recons_data(node, x_batch, l_seed):
    # type: (int, GNode, np.ndarray, np.ndarray) -> list
    with tr.no_grad():
        z_batch_pre, z_batch_post, x_recon_pre, x_recon_post = get_x_clf_plot_data(node, x_batch)
        pred_post = node.gmm_predict(as_np(z_batch_post))

        print(x_recon_post.shape)

        x_recon_post_child0 = x_recon_post[np.where(pred_post == 0)]
        l_seed_ch0 = l_seed[np.where(pred_post == 0)]
        print(x_recon_post_child0.shape)
        print(l_seed_ch0.shape)

        x_recon_post_child1 = x_recon_post[np.where(pred_post == 1)]
        l_seed_ch1 = l_seed[np.where(pred_post == 1)]
        print(l_seed_ch1.shape)
        print(x_recon_post_child1.shape)

        np.savez('x_recon_post_child0_fixed_2', recon_ch0 = x_recon_post_child0, l_seed_ch0 = l_seed_ch0)
        np.savez('x_recon_post_child1_fixed_2', recon_ch1 = x_recon_post_child1, l_seed_ch1 = l_seed_ch1)

#  node 0

phase1_epochs = 7
gan = ImgGAN.create_from_hyperparams('node0', H, '0')
means = as_np(gan.z_op_params.means)
cov = as_np(gan.z_op_params.cov)
dist_params = DistParams(means=means, cov=cov, pi=1.0, prob=1.0)

dl = DataLoaderFactory.get_dataloader(H.dataloader, H.batch_size, H.batch_size)

x_seed_full, l_seed_full = dl.random_batch('test', 512)
x_seed = x_seed_full[np.where(l_seed_full != 1)]
l_seed = l_seed_full[np.where(l_seed_full != 1)]

tree = GanTree('gtree', ImgGAN, H, x_seed)
root = tree.create_child_node(dist_params, gan)

root.set_trainer(dl, H, train_config, Model=GanImgTrainer)

# GNode.load('../experiments/mxp_1_trial/best_node-10.pt', root)
# GNode.loadwithkmeans('best_root_phase1_mnistdc.pickle', root)
root = GNode.loadwithkmeans('best_root_phase1_mnistdc_'+ str(phase1_epochs) +'.pickle', root)

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
node_id = find_next_node()

try:
    logger.info(colored('Next Node to split: %d' % node_id, 'green', attrs=['bold']))
    root = tree.nodes[node_id]
    # train_node(root, x_clf_iters=1500, gan_iters=20000)  # , min_gan_iters=5000, x_clf_lim=0.00001, x_recon_limit=0.004)
    train_node(root, x_clf_iters=14, gan_iters=20000)  # , min_gan_iters=5000, x_clf_lim=0.00001, x_recon_limit=0.004)

except Exception as e:
    pool.close()
    traceback.print_exc()
    raise Exception(e)

# # GNode.load('best_node.pickle', root)
# # for i in range(20):
# #     root.train(5000)
# #     root.save('../experiments/' + exp_name + '/best_node-' + str(i) + '.pt')

root.save('best_root_phase1_mnistdc.pickle')
root.get_child(0).save('best_child0_phase1_mnistdc.pickle')
root.get_child(1).save('best_child1_phase1_mnistdc.pickle')


# # # print('Iter: %d' % (iter_no + 1))
# # print('Training Complete.')

get_recons_data(root, x_seed, l_seed)
