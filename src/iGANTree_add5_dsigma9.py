from __future__ import print_function, division
import os, argparse, logging, json
import traceback

from termcolor import colored

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys

from tqdm import tqdm
import numpy as np

import torch as tr
from configs import Config

from scipy.spatial import distance
import random
import math

from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt


default_args_str = '-hp hyperparams/mnist_0to4_add5.py -en iGANTree_MNIST_add5_dsigma9_trial1 -t'

if Config.use_gpu:
    print('mode: GPU')


# Argument Parsing
parser = argparse.ArgumentParser()

parser.add_argument('-g', '--gpu', default=0, help='index of the gpu to be used. default: 0')
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


###### Set Experiment Context ######

from exp_context import ExperimentContext

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

for i in range(20):
    test_1 = 'node'+str(i)+'_test_above_threshold'
    test_2 = 'node'+str(i)+'_test_assigned'
    test_3 = 'node'+str(i)+'_test_confidence'
    test_4 = 'node'+str(i)+'_test_embedding_histogram'
    test_5 = 'node'+str(i)+'_test_embedding_plots'
    test_6 = 'node'+str(i)+'_test_labels_distribution'
    test_7 = 'node'+str(i)+'_test_mean_axis_histogram'
    train_1 = 'node'+str(i)+'_train_above_threshold'
    train_2 = 'node'+str(i)+'_train_assigned'
    train_3 = 'node'+str(i)+'_train_confidence'
    train_4 = 'node'+str(i)+'_train_embedding_histogram'
    train_5 = 'node'+str(i)+'_train_embedding_plots'
    train_6 = 'node'+str(i)+'_train_labels_distribution'
    train_7 = 'node'+str(i)+'_train_mean_axis_histogram'

    paths_to_make = [test_1, 
                     test_2,
                     test_3,
                     test_4, 
                     test_5,
                     test_6,
                     test_7,
                     train_1, 
                     train_2,
                     train_3,
                     train_4,
                     train_5,
                     train_6,
                     train_7]

    for path in paths_to_make:
        if not os.path.exists(Paths.results_base_dir + "/" + path):        
           os.makedirs(Paths.results_base_dir + "/" + path)
        


##### Create required directories
model_utils.setup_dirs()


##### Model and Training related imports
from dataloaders.factory import DataLoaderFactory
from base.hyperparams import Hyperparams
from trainers.gan_image_trainer import GanImgTrainer
from trainers.gan_image_trainer import save_image
from models.fashion.gan import ImgGAN
from models.toy.gt.gantree import GanTree
from models.toy.gt.gnode import GNode, KMeansCltr
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

with open(Paths.exp_hyperparams_file, "w") as fp:
    fp.write(hyperparams_string_content)

train_config = TrainConfig(
    n_step_tboard_log = H.n_step_tboard_log,
    n_step_console_log = H.n_step_console_log,
    n_step_validation = H.n_step_validation,
    n_step_save_params = H.n_step_save_params,
    n_step_visualize = H.n_step_visualize
)


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


def split_dataloader(node):
    dl = dl_set[node.id]

    train_split_index = node.split_x(dl.data['train'], Z_flag=True)
    test_split_index = node.split_x(dl.data['test'], Z_flag=True)

    index = 4 if dl.supervised else 2

    for i in node.child_ids:

        train_data = dl.data['train'][train_split_index[i]]
        test_data = dl.data['test'][test_split_index[i]]

        train_labels = dl.labels['train'][train_split_index[i]] if dl.supervised else None
        test_labels = dl.labels['test'][test_split_index[i]] if dl.supervised else None

        data_tuples = (
            train_data,
            test_data,
            train_labels,
            test_labels
        )

        print(train_data.shape)
        print(test_data.shape)
        print(train_labels.shape)
        print(test_labels.shape)

        dl_set[i] = CustomDataLoader.create_from_parent(dl, data_tuples[:index])


def get_embeddings(node, split):

    if split == 'train':
        data = dl_set[node.id].data['train']

    Z = node.post_gmm_encode(data)

    return Z


def visualize_embeddings(node, split, threshold, iter_no, phase = None):
    with tr.no_grad():
        if split == 'train':
            data = dl_set[node.id].data[split]
            labels = dl_set[node.id].labels[split]
        elif split == 'test':
            data = x_seed
            labels = l_seed

        Z = node.post_gmm_encode(data)

        pca_z = PCA(n_components = 2)

        z_transformed = pca_z.fit_transform(Z)

        color = ['r', 'b', 'g']
        colors = [color[int(x)] for x in labels]

        b = 20
        fig = plt.figure(figsize=(6.5, 6.5))

        ax = fig.add_subplot(111)
        ax.set_xlim(-b, b)
        ax.set_ylim(-b, b)

        ax.scatter(z_transformed[:, 0], z_transformed[:, 1], s = 0.5, c = colors)


        node.trainer.writer[split].add_figure(node.name + '_' + phase +'_plots', fig, iter_no)

        path = Paths.get_result_path(node.name + '_' + split + '_embedding_plots/'+ phase + '_plot_%03d' % (iter_no))
        fig.savefig(path)
        plt.close(fig)


def visualize_images(node, split, iter_no, phase):
    with tr.no_grad():
        if split == 'train':
            data = dl_set[node.id].data['train'][:1000]
            preds = node.kmeans.pred[:1000]
        elif split == 'test':
            data = x_seed
            preds = l_seed.cpu().numpy()

        x_data_child0 = data[np.where(preds == 0)].cuda()
        x_data_child1 = data[np.where(preds == 1)].cuda()

        if x_data_child0.shape[0] > 64:
            x_data_child0 = x_data_child0[:64]

        if x_data_child1.shape[0] > 64:
            x_data_child1 = x_data_child1[:64]

        if x_data_child0.shape[0] == 0:
            x_data_child0 = node.trainer.seed_data[split]['x']

        if x_data_child1.shape[0] == 0:
            x_data_child1 = node.trainer.seed_data[split]['x']

        z_data_child0 = node.get_child(0).gan.sample((x_data_child0.shape[0],))
        z_data_child1 = node.get_child(1).gan.sample((x_data_child1.shape[0],))

        x_recon_child0 = node.get_child(0).gan.reconstruct_x(x_data_child0)
        x_recon_child1 = node.get_child(1).gan.reconstruct_x(x_data_child1)
        x_gen_child0 = node.get_child(0).gan.decode(z_data_child0)
        x_gen_child1 = node.get_child(1).gan.decode(z_data_child1)

        recon_img_child0 = save_image(x_recon_child0)
        gen_img_child0 = save_image(x_gen_child0)
        recon_img_child1 = save_image(x_recon_child1)
        gen_img_child1 = save_image(x_gen_child1)
        real_img_child0 = save_image(x_data_child0)
        real_img_child1 = save_image(x_data_child1)

        node.trainer.writer[split].add_image(node.name + '_' + phase + '_child0_recon', recon_img_child0[0], iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child0_gen', gen_img_child0[0], iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child0_real', real_img_child0[0], iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child1_recon', recon_img_child1[0], iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child1_gen', gen_img_child1[0], iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child1_real', real_img_child1[0], iter_no)


def z_histogram_plot(node, split, iter_no, phase):
    with tr.no_grad():
        if split == 'train':
            data = dl_set[node.id].data[split]
        elif split == 'test':
            data = x_seed

        Z = node.post_gmm_encode(data)

        for i in range(Z.shape[1]):
            plot_data = Z[:, i]
            plt.hist(plot_data)

            fig_histogram = plt.gcf()
            node.trainer.writer[split].add_histogram(node.name + '_' + phase + '_embedding_' + str(i), plot_data, iter_no)
            path_embedding_hist = Paths.get_result_path(node.name + '_' + split +  '_embedding_histogram/' + phase + 'embedding_%03d_%01d' % (iter_no, i))
            fig_histogram.savefig(path_embedding_hist)
            plt.close(fig_histogram)


def get_labels_distribution(node, split):
    iter_no = 0
    no_of_classes = H.no_of_classes
    with tr.no_grad():

        if split == 'train':
            data = dl_set[node.id].data[split]
            labels = dl_set[node.id].labels[split]
        elif split == 'test':
            data = x_seed
            labels = l_seed

        Z = node.post_gmm_encode(data)
        
        pred = node.gmm_predict(Z)

        labels_ch0 = labels[np.where(pred == 0)]
        labels_ch1 = labels[np.where(pred == 1)]

        np.savez(node.name + '_' + split + '_child_labels', labels_ch0 = labels_ch0, labels_ch1 = labels_ch1)

        count_ch0 = [0 for i in range(no_of_classes)]
        count_ch1 = [0 for i in range(no_of_classes)]
        prob_ch0 = [0 for i in range(no_of_classes)]
        prob_ch1 = [0 for i in range(no_of_classes)] 

        for i in labels_ch0:
            count_ch0[i] += 1

        for i in labels_ch1:
            count_ch1[i] += 1

        for i in range(no_of_classes):
            if (count_ch0[i] + count_ch1[i]) != 0:
                prob_ch0[i] = count_ch0[i] * 1.0 / (count_ch0[i] + count_ch1[i])
                prob_ch1[i] = count_ch1[i] * 1.0 / (count_ch0[i] + count_ch1[i])
            else:
                prob_ch0[i] = 0
                prob_ch1[i] = 0

        barWidth = 0.3
        r1 = np.arange(len(count_ch0))
        r2 = [x+barWidth for x in r1]

        plt.bar(r1, prob_ch0, width = barWidth, color = 'red', edgecolor = 'black', capsize=7)
        plt.bar(r2, prob_ch1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7)
        plt.xticks([r + barWidth for r in range(len(prob_ch0))], [str(i) for i in range(no_of_classes)])
        plt.ylabel('percentage')

        fig_labels_prob = plt.gcf()
        node.trainer.writer[split].add_figure(node.name + '_labels_prob', fig_labels_prob, iter_no)
        path_labels_prob = Paths.get_result_path(node.name + '_' + split + '_labels_distribution/probability_%03d' % (iter_no))
        fig_labels_prob.savefig(path_labels_prob)
        plt.close(fig_labels_prob)


        plt.bar(r1, count_ch0, width = barWidth, color = 'red', edgecolor = 'black', capsize=7)
        plt.bar(r2, count_ch1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7)
        plt.xticks([r + barWidth for r in range(len(count_ch0))], [str(i) for i in range(no_of_classes)])
        plt.ylabel('count')

        fig_labels_count = plt.gcf()
        node.trainer.writer[split].add_figure(node.name + '_labels_distribution', fig_labels_count, iter_no)
        path_labels_count = Paths.get_result_path(node.name + '_' + split +  '_labels_distribution/count_%03d' % (iter_no))
        fig_labels_count.savefig(path_labels_count)
        plt.close(fig_labels_count)




def plot_cluster_graphs(node, split, threshold, iter_no, phase):
    no_of_classes = H.no_of_classes

    with tr.no_grad():
        if split == 'train':
            data = dl_set[node.id].data[split]
            labels = dl_set[node.id].labels[split]
        elif split == 'test':
            data = x_seed
            labels = l_seed

        Z = node.post_gmm_encode(data)

        if split == 'train':
            p = node.kmeans.pred
        elif split == 'test':
            p = l_seed.cpu().numpy()

        """ plot the count of unassigned vs assigned labels
            purple -- unassigned
            green -- assigned """

        unassigned_labels = [0 for i in range(no_of_classes)]
        assigned_labels = [0 for i in range(no_of_classes)]

        for i in range(len(p)):
            if p[i] == 2:
                unassigned_labels[labels[i]] += 1
            else:
                assigned_labels[labels[i]] += 1

        barWidth = 0.3
        r1 = np.arange(len(unassigned_labels))
        r2 = [x+barWidth for x in r1]

        plt.bar(r1, unassigned_labels, width = barWidth, color = 'purple', edgecolor = 'black', capsize=7)
        plt.bar(r2, assigned_labels, width = barWidth, color = 'green', edgecolor = 'black', capsize=7)
        plt.xticks([r + barWidth for r in range(len(unassigned_labels))], [str(i) for i in range(no_of_classes)])
        plt.ylabel('count')

        fig_assigned = plt.gcf()
        node.trainer.writer[split].add_figure(node.name + '_' + phase + '_assigned_labels_count', fig_assigned, iter_no)
        path_assign = Paths.get_result_path(node.name + '_' + split +  '_assigned/' + phase + 'assigned_%03d' % (iter_no))
        fig_assigned.savefig(path_assign)
        plt.close(fig_assigned)



        """ plot the percentage of assigned labels in cluster 0 and cluster 1
            red -- cluster 0
            blue -- cluster 1 """

        l_seed_ch0 = labels[np.where(p == 0)]
        l_seed_ch1 = labels[np.where(p == 1)]

        count_ch0 = [0 for i in range(no_of_classes)]
        count_ch1 = [0 for i in range(no_of_classes)]
        prob_ch0 = [0 for i in range(no_of_classes)]
        prob_ch1 = [0 for i in range(no_of_classes)]

        for i in l_seed_ch0:
            count_ch0[i] += 1

        for i in l_seed_ch1:
            count_ch1[i] += 1

        for i in range(no_of_classes):
            if (count_ch0[i] + count_ch1[i]) != 0:
                prob_ch0[i] = count_ch0[i] * 1.0 / (count_ch0[i] + count_ch1[i])
                prob_ch1[i] = count_ch1[i] * 1.0 / (count_ch0[i] + count_ch1[i])
            else:
                prob_ch0[i] = 0
                prob_ch1[i] = 0

        plt.bar(r1, prob_ch0, width = barWidth, color = 'red', edgecolor = 'black', capsize=7)
        plt.bar(r2, prob_ch1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7)
        plt.xticks([r + barWidth for r in range(len(prob_ch0))], [str(i) for i in range(no_of_classes)])
        plt.ylabel('percentage')

        fig_confidence = plt.gcf()
        node.trainer.writer[split].add_figure(node.name +  '_' + phase + '_confidence', fig_confidence, iter_no)
        path_confidence = Paths.get_result_path(node.name + '_' + split + '_confidence/' + phase + 'confidence_%03d' % (iter_no))
        fig_confidence.savefig(path_confidence)
        plt.close(fig_confidence)


        """ get count of points that exceed the threshold of phase 1 part 2 """

        aboveThresholdLabels_ch0 = [0 for i in range(no_of_classes)]
        aboveThresholdLabels_ch1 = [0 for i in range(no_of_classes)]

        for i in range(len(p)):
            if p[i] == 0:
                if (distance.mahalanobis(Z[i], node.kmeans.means[0], node.kmeans.covs[0])) > threshold:
                    aboveThresholdLabels_ch0[labels[i]] += 1
            elif p[i] == 1:
                if (distance.mahalanobis(Z[i], node.kmeans.means[1], node.kmeans.covs[1])) > threshold:
                    aboveThresholdLabels_ch1[labels[i]] += 1

        plt.bar(r1, aboveThresholdLabels_ch0, width = barWidth, color = 'red', edgecolor = 'black', capsize=7)
        plt.bar(r2, aboveThresholdLabels_ch1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7)
        plt.xticks([r + barWidth for r in range(len(aboveThresholdLabels_ch0))], [str(i) for i in range(no_of_classes)])
        plt.ylabel('count')

        fig_above_threshold = plt.gcf()
        node.trainer.writer[split].add_figure(node.name + '_' + phase + '_above_threshold', fig_above_threshold, iter_no)
        path_above_threshold = Paths.get_result_path(node.name + '_' + split + '_above_threshold/' + phase + '%03d' % (iter_no))
        fig_above_threshold.savefig(path_above_threshold)
        plt.close(fig_above_threshold)



def plot_mean_axis_distribution(node, split, iter_no, phase):

    mean0 = node.kmeans.means[0]
    mean1 = node.kmeans.means[1]

    direction = (mean1 - mean0) / np.linalg.norm(mean1 - mean0)

    if split == 'train':
        data = dl_set[node.id].data['train']
    elif split == 'test':
        data = x_seed

    Z = node.post_gmm_encode(data)

    projection = np.zeros(Z.shape)

    for j in range(Z.shape[0]):
        projection[j] = mean0 + direction * np.dot(Z[j] - mean0, direction)

    for i in range(projection.shape[1]):
        plot_data_tensorboard = projection[:, i] 
        plot_data = [projection[:, i], mean0[i], mean1[i]]
        plt.hist(plot_data, color = ['g', 'r', 'b'])

        fig_mean_axis_histogram = plt.gcf()
        node.trainer.writer[split].add_histogram(node.name + '_' + phase + '_mean_axis_' + str(i), plot_data_tensorboard, iter_no)
        path_mean_axis_hist = Paths.get_result_path(node.name + '_' + split +  '_mean_axis_histogram/' + phase + '%03d_%01d' % (iter_no, i))
        fig_mean_axis_histogram.savefig(path_mean_axis_hist)
        plt.close(fig_mean_axis_histogram)



## phase 1 training, with labels already assigned

def train_phase_1(node, threshold = 2.5, phase1_epochs = 7):
    print('entered phase 1')

    print(distance.mahalanobis(node.kmeans.means[1], node.kmeans.means[0], node.kmeans.covs[0]))
    print(threshold)

    """ define the train dataset and the corresponding labels"""

    train_data = dl_set[node.id].train_data()
    train_data_labels = dl_set[node.id].train_data_labels()

    """ visualize the initial plot """

    # pca2rand = visualize_plots_phase1part2(iter_no=0, node=node, x_batch=x_seed, labels=l_seed, tag='x_clf_plots', threshold = threshold)


    """ define the lists of data for loading into the dataloader """

    training_list = []
    assigned_index = []
    unassigned_index = []


    """ load the node , i.e. checkpoints """

    # node = GNode.loadwithkmeans('best_root_phase1_mnistdc_'+ str(phase1_epochs) +'.pickle', node)
    # child0 = GNode.loadwithkmeans('best_child0_phase1_mnistdc_'+ str(phase1_epochs) + '.pickle', node.get_child(0))
    # child1 = GNode.loadwithkmeans('best_child1_phase1_mnistdc_'+ str(phase1_epochs) + '.pickle', node.get_child(1))
    # node.child_nodes = {1: child0, 2: child1}


    """ train mode separation algorithm (phase 1) """

    for j in range(0, phase1_epochs):

        print("epoch number: " + str(j))

        """ get encodings of train data """

        Z = get_embeddings(node = node, split = 'train')


        """ get batch size of training for the dataloader """

        batchSize = dl_set[node.id].batch_size['train']

        if len(Z) % batchSize == 0:
            n_iterations = (len(Z) // batchSize) 
        else:
            n_iterations = (len(Z) // batchSize) + 1


        """ save the node, ie checkpoints, every epoch """

        # if j == reassignment_epoch:

        #     node.save('best_root_phase1_mnistdc_whole_'+ str(j) +'.pickle')
        #     node.get_child(0).save('best_child0_phase1_mnistdc_whole_'+ str(j) + '.pickle')
        #     node.get_child(1).save('best_child1_phase1_mnistdc_whole_' + str(j) + '.pickle')


        """ start training """

        with tqdm(total=n_iterations) as pbar:
            for iter_no in range(n_iterations):


                """ define the current total iteration number """

                current_iter_no = j * n_iterations + iter_no
                node.trainer.iter_no = current_iter_no


                """ define the training data for a particular batch """
                
                p = node.kmeans.pred

                assigned_index = np.where(p != 2)[0]
                length_assi = len(assigned_index)

                unassigned_index = np.where(p == 2)[0]
                length_unassi = len(unassigned_index)


                if length_assi == 0 or length_unassi == 0:

                    """ if no assigned or unaasigned labels, take a normal batch """

                    start_no = (iter_no * batchSize) % len(train_data)
                    end_no = (start_no + batchSize) % len(train_data)

                    if end_no < start_no:
                        training_list = [i for i in range(start_no, len(train_data))]
                        training_list.extend([i for i in range(0, end_no)])
                    else:
                        training_list = [i for i in range(start_no, end_no)]

                elif (batchSize/2 >= length_assi) or (batchSize/2 >= length_unassi):

                    """ if number of assigned or unassigned labels is below half the batch size,
                    take equal amount of assigned and unasssigned labels till the size of 
                    the batch exceeds the batch size. """

                    training_list = random.sample(list(assigned_index), min(length_unassi, length_assi))
                    training_list.extend(random.sample(list(unassigned_index), min(length_assi, length_unassi)))

                    while(len(training_list) < batchSize):
                        training_list.extend(random.sample(list(assigned_index), min(length_unassi, length_assi)))
                        training_list.extend(random.sample(list(unassigned_index), min(length_assi, length_unassi)))

                else:

                    """ take an equal number of assigned and unassigned labels, sequentially """

                    """ take the assigned labels """

                    start_no_assi = int((iter_no * batchSize / 2) % length_assi)
                    end_no_assi = int((start_no_assi + batchSize/2) % length_assi)

                    if end_no_assi < start_no_assi:
                        training_list = [i for i in assigned_index[start_no_assi:]]
                        training_list.extend([i for i in assigned_index[:end_no_assi]])
                    else:
                        training_list = [i for i in assigned_index[start_no_assi:end_no_assi]]

                    """ take the unassigned labels """

                    start_no_unassi = int((iter_no * batchSize/2) % length_unassi)
                    end_no_unassi = int((start_no_unassi + batchSize/2) % length_unassi)

                    if end_no_unassi < start_no_unassi:
                        training_list.extend([i for i in unassigned_index[start_no_unassi:]])
                        training_list.extend([i for i in unassigned_index[:end_no_unassi]])
                    else:
                        training_list.extend([i for i in unassigned_index[start_no_unassi:end_no_unassi]])

                
                """ get the training batch """

                x_clf_train_batch = train_data[training_list].cuda()


                """ train the node """

                z_batch, x_recon, preds, x_clf_loss_assigned, x_assigned_recon_loss, loss_assigned, x_clf_loss_unassigned, x_unassigned_recon_loss, loss_unassigned, x_clf_cross_loss, loss_recon = node.step_train_x_clf_phase1(x_clf_train_batch, training_list, with_PCA = False, threshold = threshold)


                """ plot the losses and ratio of labels and unlabeled points every 10 iterations"""

                if current_iter_no % 10 == 0:

                    """ calculate the ratio of assigned points in cluster 0
                        calculate the ratio of points not assigned """

                    positive = np.sum(preds == 0.)
                    negative = np.sum(preds == 1.)
                    neutral = np.sum(preds == 2.)

                    ratio = positive * 1.0 / (positive + negative)
                    unassigned_ratio = neutral * 1.0 / len(preds)


                    """ plot on tensorboard """

                    node.trainer.writer['train'].add_scalar(node.name + '_x_clf_loss_assigned', x_clf_loss_assigned, current_iter_no)
                    node.trainer.writer['train'].add_scalar(node.name + '_x_assigned_recon_loss', x_assigned_recon_loss, current_iter_no)
                    node.trainer.writer['train'].add_scalar(node.name + '_loss_assigned', loss_assigned, current_iter_no)
                    node.trainer.writer['train'].add_scalar(node.name + '_split_ratio', ratio, current_iter_no)
                    node.trainer.writer['train'].add_scalar(node.name + '_x_clf_cross_loss', x_clf_cross_loss, current_iter_no)
                    node.trainer.writer['train'].add_scalar(node.name + '_unassigned_ratio', unassigned_ratio, current_iter_no)
                    node.trainer.writer['train'].add_scalar(node.name + '_x_clf_loss_unassigned', x_clf_loss_unassigned, current_iter_no)
                    node.trainer.writer['train'].add_scalar(node.name + '_x_unassigned_recon_loss', x_unassigned_recon_loss, current_iter_no)
                    node.trainer.writer['train'].add_scalar(node.name + '_loss_unassigned', loss_unassigned, current_iter_no)
                    node.trainer.writer['train'].add_scalar(node.name + '_loss_recon', loss_recon, current_iter_no)
                  

                """ update the labels of unassigned labels of training batch if the encodings 
                    are inside the threshold """

                node.updatePredictions(x_clf_train_batch, training_list, threshold)



                """ show figures of training data every 200 iterations"""

                if current_iter_no % 500 == 0:
                    
                    visualize_images(node, 'train', current_iter_no, 'phase_1_2')
                    visualize_images(node, 'test', current_iter_no, 'phase_1_2')
                    visualize_embeddings(node, 'train', threshold, current_iter_no, 'phase_1_2')
                    visualize_embeddings(node, 'test', threshold, current_iter_no, 'phase_1_2')
                    z_histogram_plot(node, 'train', current_iter_no, 'phase_1_2')
                    z_histogram_plot(node, 'test', current_iter_no, 'phase_1_2')
                    plot_cluster_graphs(node, 'train', threshold, current_iter_no, 'phase_1_2')
                    plot_cluster_graphs(node, 'test', threshold, current_iter_no, 'phase_1_2')
                    plot_mean_axis_distribution(node, 'train', current_iter_no, 'phase_1_2')
                    plot_mean_axis_distribution(node, 'test', current_iter_no, 'phase_1_2')


                """ show validation plots every 40 iterations (similar to the training plots) """

                if current_iter_no % 500 == 0:

                    preds_test, x_clf_loss_assigned_test, x_assigned_recon_loss_test, loss_assigned_test, x_clf_loss_unassigned_test, x_unassigned_recon_loss_test, loss_unassigned_test, x_clf_cross_loss_test, loss_recon_test = node.step_predict_test(x_seed, with_PCA = True, threshold = threshold)

                    positive_test = np.sum(preds_test == 0.)
                    negative_test = np.sum(preds_test == 1.)
                    neutral_test = np.sum(preds_test == 2.)

                    ratio_test = positive_test * 1.0 / (positive_test + negative_test)
                    unassigned_ratio_test = neutral_test * 1.0 / len(preds_test)

                    node.trainer.writer['test'].add_scalar(node.name + '_x_clf_loss_assigned', x_clf_loss_assigned_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar(node.name + '_x_assigned_recon_loss', x_assigned_recon_loss_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar(node.name + '_loss_assigned', loss_assigned_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar(node.name + '_split_ratio', ratio_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar(node.name + '_x_clf_cross_loss', x_clf_cross_loss_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar(node.name + '_unassigned_ratio', unassigned_ratio_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar(node.name + '_x_clf_loss_unassigned', x_clf_loss_unassigned_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar(node.name + '_x_unassigned_recon_loss', x_unassigned_recon_loss_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar(node.name + '_loss_unassigned', loss_unassigned_test, current_iter_no)
                    node.trainer.writer['test'].add_scalar(node.name + '_loss_recon', loss_recon_test, current_iter_no)

                pbar.update(n=1)


## phase 2 training (GAN training of both child nodes)

def train_phase_2(node, n_iterations, incremental_nodes_branch):
    node.get_child(0).trainer.iter_no = 0 
    node.get_child(1).trainer.iter_no = 0 

    if node.get_child(0).id not in incremental_nodes_branch:
        for param in node.get_child(0).gan.generator.parameters():
            param.requires_grad = False   

        for param in node.get_child(0).gan.discriminator.parameters():
            param.requires_grad = False

        node.get_child(0).trainer.train_discriminator = False
    elif node.get_child(1).id not in incremental_nodes_branch:
        for param in node.get_child(1).gan.generator.parameters():
            param.requires_grad = False   

        for param in node.get_child(1).gan.discriminator.parameters():
            param.requires_grad = False

        node.get_child(1).trainer.train_discriminator = False


    with tqdm(total=n_iterations) as pbar:
        for iter_no in range(n_iterations):
            for i in node.child_ids:
                node.child_nodes[i].trainer.train(1, enable_tqdm = False)
                pbar.update(n=0.5)

    z_histogram_plot(node.get_child(0), 'train', iter_no, 'phase_2')
    z_histogram_plot(node.get_child(0), 'test', iter_no, 'phase_2')

    z_histogram_plot(node.get_child(1), 'train', iter_no, 'phase_2')
    z_histogram_plot(node.get_child(1), 'test', iter_no, 'phase_2')


## Seeking algorithm

def get_branch_incremental(new_class_dl, inc_threshold):
    incremental_nodes = [0]
    within_threshold = True
    incremental_node_id = 0

    while(within_threshold):
        probab_inc_node = tree.nodes[incremental_node_id]

        if len(probab_inc_node.child_ids) == 0:
            break

        inc_Z = probab_inc_node.post_gmm_encode(new_class_dl.data['train'])
        total_inc_Z_distance_child0 = 0
        total_inc_Z_distance_child1 = 0

        for i in range(len(inc_Z)):
            total_inc_Z_distance_child0 += distance.mahalanobis(inc_Z[i], probab_inc_node.kmeans.means[0], probab_inc_node.kmeans.covs[0])
            total_inc_Z_distance_child1 += distance.mahalanobis(inc_Z[i], probab_inc_node.kmeans.means[1], probab_inc_node.kmeans.covs[1])

        avg_inc_Z_distance_child0 = total_inc_Z_distance_child0 / len(inc_Z)
        avg_inc_Z_distance_child1 = total_inc_Z_distance_child1 / len(inc_Z)

        if (avg_inc_Z_distance_child0 < avg_inc_Z_distance_child1) and (avg_inc_Z_distance_child0 <= inc_threshold):
            incremental_nodes.append(probab_inc_node.child_ids[0])
            incremental_node_id = probab_inc_node.child_ids[0]
        elif (avg_inc_Z_distance_child0 > avg_inc_Z_distance_child1) and (avg_inc_Z_distance_child1 <= inc_threshold):
            incremental_nodes.append(probab_inc_node.child_ids[1])
            incremental_node_id = probab_inc_node.child_ids[1]
        else:
            within_threshold = False

    return incremental_nodes


## get leaf nodes of a particular node

def get_leaf_nodes(node):
    if len(node.child_ids) == 0:
        return [node.id]
    else:
        leaf_child0 = get_leaf_nodes(node.get_child(0))
        leaf_child1 = get_leaf_nodes(node.get_child(1))

        leaf_nodes = leaf_child0 + leaf_child1

        return leaf_nodes


## generate samples of particular node

def decode_samples(node, X, batch = 128):
    X = X.cuda()
    Z = []
    n_batches = (X.shape[0] + batch - 1) // batch
    for i in range(n_batches):
        x = X[i * batch:(i + 1) * batch]
        z = node.gan.decode(x)
        Z.append(z)
    Z = tr.from_numpy(np.concatenate(Z))
    return Z


## generate data-loader for ancestor class

def get_new_inc_dataloader(inc_node, new_class_dl, total_inc_train_examples, total_inc_test_examples, incremental_nodes_branch):

    ## get leaf nodes of subtree having root as the particular node to be trained
    inc_node_leaf_nodes = get_leaf_nodes(inc_node)
    inc_node_leaf_nodes_child0 = get_leaf_nodes(inc_node.get_child(0))
    inc_node_leaf_nodes_child1 = get_leaf_nodes(inc_node.get_child(1))

    print(inc_node_leaf_nodes)
    print(inc_node_leaf_nodes_child0)
    print(inc_node_leaf_nodes_child1)

    ## get proportion of examples for each node

    inc_node_leaf_nodes_prior_prob = {}
    train_examples = {}
    test_examples = {}

    for i in range(len(inc_node_leaf_nodes)):
        inc_node_leaf_nodes_prior_prob[inc_node_leaf_nodes[i]] = tree.nodes[inc_node_leaf_nodes[i]].prob
        
    total_prior_prob = sum(inc_node_leaf_nodes_prior_prob.itervalues())

    for i in range(len(inc_node_leaf_nodes)):
        inc_node_leaf_nodes_prior_prob[inc_node_leaf_nodes[i]] /= total_prior_prob

    print(inc_node_leaf_nodes_prior_prob)


    ## get total number of samples for each class

    for i in range(len(inc_node_leaf_nodes) - 1):
        train_examples[inc_node_leaf_nodes[i]] = int(total_inc_train_examples * inc_node_leaf_nodes_prior_prob[inc_node_leaf_nodes[i]]) 
        test_examples[inc_node_leaf_nodes[i]] = int(total_inc_test_examples * inc_node_leaf_nodes_prior_prob[inc_node_leaf_nodes[i]]) 

    train_examples[inc_node_leaf_nodes[-1]] = total_inc_train_examples - sum(train_examples.itervalues())
    test_examples[inc_node_leaf_nodes[-1]] = total_inc_test_examples - sum(test_examples.itervalues())

    print(train_examples)
    print(test_examples)


    ## generate samples from leaf nodes

    node_to_sample = tree.nodes[inc_node_leaf_nodes[0]]
    train_data = decode_samples(node_to_sample, node_to_sample.gan.sample((train_examples[inc_node_leaf_nodes[0]],)))
    test_data = decode_samples(node_to_sample, node_to_sample.gan.sample((test_examples[inc_node_leaf_nodes[0]],)))

    if node_to_sample.id in inc_node_leaf_nodes_child0:
        train_preds = tr.LongTensor([0 for i in range(len(train_data))])
        test_preds = tr.LongTensor([0 for i in range(len(test_data))])
    elif node_to_sample.id in inc_node_leaf_nodes_child1:
        train_preds = tr.LongTensor([1 for i in range(len(train_data))])
        test_preds = tr.LongTensor([1 for i in range(len(test_data))])

    for i in range(1, len(inc_node_leaf_nodes)):
        node_to_sample = tree.nodes[inc_node_leaf_nodes[i]]


        train_data = tr.cat((train_data, decode_samples(node_to_sample, node_to_sample.gan.sample((train_examples[inc_node_leaf_nodes[i]],)))), 0)
        test_data = tr.cat((test_data, decode_samples(node_to_sample, node_to_sample.gan.sample((test_examples[inc_node_leaf_nodes[i]],)))), 0)

        if node_to_sample.id in inc_node_leaf_nodes_child0:
            train_preds = tr.cat((train_preds, tr.LongTensor([0 for t in range(train_examples[inc_node_leaf_nodes[i]])])), 0)
            test_preds = tr.cat((test_preds, tr.LongTensor([0 for t in range(test_examples[inc_node_leaf_nodes[i]])])), 0)
        elif node_to_sample.id in inc_node_leaf_nodes_child1:
            train_preds = tr.cat((train_preds, tr.LongTensor([1 for t in range(train_examples[inc_node_leaf_nodes[i]])])), 0)
            test_preds = tr.cat((test_preds, tr.LongTensor([1 for t in range(test_examples[inc_node_leaf_nodes[i]])])), 0)


    ## include samples of new class

    train_data = tr.cat((train_data, new_class_dl.data['train']), 0)
    test_data = tr.cat((test_data, new_class_dl.data['test']), 0)

    if inc_node.get_child(0).id in incremental_nodes_branch:
        train_preds = tr.cat((train_preds, tr.LongTensor([0 for i in range(len(new_class_dl.data['train']))])), 0)
        test_preds = tr.cat((test_preds, tr.LongTensor([0 for i in range(len(new_class_dl.data['test']))])), 0)
    elif inc_node.get_child(1).id in incremental_nodes_branch:
        train_preds = tr.cat((train_preds, tr.LongTensor([1 for i in range(len(new_class_dl.data['train']))])), 0)
        test_preds = tr.cat((test_preds, tr.LongTensor([1 for i in range(len(new_class_dl.data['test']))])), 0)

    train_preds = train_preds.type(tr.LongTensor)
    test_preds = test_preds.type(tr.LongTensor)


    ## create new dataloader

    data_tuples = (
        train_data,
        test_data,
        train_preds,
        test_preds
    )

    print(train_data.shape)
    print(test_data.shape)
    print(train_preds.shape)
    print(test_preds.shape)

    inc_node.kmeans.pred = train_preds.numpy()

    dl_inc_new = CustomDataLoader.create_from_parent(dl_set[inc_node.id], data_tuples)

    dl_inc_new.shuffle('train')
    dl_inc_new.shuffle('test')

    inc_node.kmeans.pred = dl_inc_new.labels['train'].numpy()

    return dl_inc_new


## generate data-loader for new parent class

def generate_new_parent_dl(new_parent_node, new_class_dl, child_number, total_inc_train_examples, total_inc_test_examples):
    ## get leaf nodes of subtree having root as the particular node to be trained
    inc_node_leaf_nodes = get_leaf_nodes(new_parent_node.get_child(child_number))

    print(inc_node_leaf_nodes)

    ## get proportion of examples for each node

    inc_node_leaf_nodes_prior_prob = {}
    train_examples = {}
    test_examples = {}

    for i in range(len(inc_node_leaf_nodes)):
        inc_node_leaf_nodes_prior_prob[inc_node_leaf_nodes[i]] = tree.nodes[inc_node_leaf_nodes[i]].prob
        
    total_prior_prob = sum(inc_node_leaf_nodes_prior_prob.itervalues())

    for i in range(len(inc_node_leaf_nodes)):
        inc_node_leaf_nodes_prior_prob[inc_node_leaf_nodes[i]] /= total_prior_prob

    print(inc_node_leaf_nodes_prior_prob)

    ## get total number of samples for each class

    for i in range(len(inc_node_leaf_nodes) - 1):
        train_examples[inc_node_leaf_nodes[i]] = int(total_inc_train_examples * inc_node_leaf_nodes_prior_prob[inc_node_leaf_nodes[i]]) 
        test_examples[inc_node_leaf_nodes[i]] = int(total_inc_test_examples * inc_node_leaf_nodes_prior_prob[inc_node_leaf_nodes[i]]) 

    train_examples[inc_node_leaf_nodes[-1]] = total_inc_train_examples - sum(train_examples.itervalues())
    test_examples[inc_node_leaf_nodes[-1]] = total_inc_test_examples - sum(test_examples.itervalues())

    print(train_examples)
    print(test_examples)


    ## generate samples from leaf nodes

    node_to_sample = tree.nodes[inc_node_leaf_nodes[0]]
    train_data = decode_samples(node_to_sample, node_to_sample.gan.sample((train_examples[inc_node_leaf_nodes[0]],)))
    test_data = decode_samples(node_to_sample, node_to_sample.gan.sample((test_examples[inc_node_leaf_nodes[0]],)))
    train_preds = tr.LongTensor([child_number for i in range(len(train_data))])
    test_preds = tr.LongTensor([child_number for i in range(len(test_data))])
    
    for i in range(1, len(inc_node_leaf_nodes)):
        node_to_sample = tree.nodes[inc_node_leaf_nodes[i]]

        train_data = tr.cat((train_data, decode_samples(node_to_sample, node_to_sample.gan.sample((train_examples[inc_node_leaf_nodes[i]],)))), 0)
        test_data = tr.cat((test_data, decode_samples(node_to_sample, node_to_sample.gan.sample((test_examples[inc_node_leaf_nodes[i]],)))), 0)

        train_preds = tr.cat((train_preds, tr.LongTensor([child_number for t in range(train_examples[inc_node_leaf_nodes[i]])])), 0)
        test_preds = tr.cat((test_preds, tr.LongTensor([child_number for t in range(test_examples[inc_node_leaf_nodes[i]])])), 0)
        
    ## include samples of new class

    train_data = tr.cat((train_data, new_class_dl.data['train']), 0)
    test_data = tr.cat((test_data, new_class_dl.data['test']), 0)
    train_preds = tr.cat((train_preds, tr.LongTensor([1-child_number for i in range(len(new_class_dl.data['train']))])), 0)
    test_preds = tr.cat((test_preds, tr.LongTensor([1-child_number for i in range(len(new_class_dl.data['test']))])), 0)
    

    train_preds = train_preds.type(tr.LongTensor)
    test_preds = test_preds.type(tr.LongTensor)

    ## create new dataloader

    data_tuples = (
        train_data,
        test_data,
        train_preds,
        test_preds
    )

    print(train_data.shape)
    print(test_data.shape)
    print(train_preds.shape)
    print(test_preds.shape)

    # new_parent_node.kmeans.pred = train_preds.numpy()

    dl_inc_new = CustomDataLoader(img_size=dl_set[0].img_size,
                                 latent_size=dl_set[0].latent_size,
                                 train_batch_size=dl_set[0].batch_size['train'],
                                 test_batch_size=dl_set[0].batch_size['test'],
                                 supervised= len(data_tuples) == 4,
                                 get_data= lambda: data_tuples)

    dl_set[new_parent_node.id] = dl_inc_new

    dl_inc_new.shuffle('train')
    dl_inc_new.shuffle('test')

    # new_parent_node.kmeans.pred = dl_inc_new.labels['train'].numpy()

    return dl_inc_new


def save_nodes(node, phase):
    node.save('../experiments/'+ exp_name + '/weights/best_node_' + str(node.id) + '_mnist_' + str(phase)+'.pickle')
    node.child_nodes[node.child_ids[0]].save('../experiments/'+ exp_name + '/weights/best_node_' + str(node.id) + '_child_0_mnist_' + str(phase) + '.pickle')
    node.child_nodes[node.child_ids[1]].save('../experiments/'+ exp_name + '/weights/best_node_' + str(node.id) + '_child_1_mnist_' + str(phase) + '.pickle')



## setup initial root node

gan = ImgGAN.create_from_hyperparams('node0', H, cov_sign = '0')
means = as_np(gan.z_op_params.means)
cov = as_np(gan.z_op_params.cov)
dist_params = DistParams(means=means, cov=cov, pi=1.0, prob=1.0)

dl = DataLoaderFactory.get_dataloader(H.dataloader, H.img_size, H.batch_size, H.batch_size, H.classes)

x_seed, l_seed = dl.random_batch('test', 512)

tree = GanTree('gtree', ImgGAN, H, x_seed)
root = tree.create_child_node(dist_params, gan)
root.set_trainer(dl, H, train_config, Model=GanImgTrainer)


dl_set = {0: dl}
leaf_nodes = {0}

bash_utils.create_dir(Paths.weight_dir_path(''), log_flag=False)


## load whole gan_tree

for i_modes in range(3):
    node_id = i_modes
    load_node = tree.nodes[node_id]

    load_node = GNode.load('../experiments/GANTree_MNIST_0to4_trial1/weights/best_node_'+ str(node_id) +'_mnist_phase_1_2.pickle', load_node)

    x_seed, l_seed = dl_set[node_id].random_batch('test', 512)

    child_nodes = tree.load_children(load_node, '../experiments/GANTree_MNIST_0to4_trial1/weights/best_node_' + str(load_node.id) + '_child_')

    split_dataloader(load_node)

    load_node.prob = len(dl_set[node_id].data['train'])/ len(dl_set[0].data['train'])
    load_node.get_child(0).prob = load_node.prob * (len(dl_set[child_nodes[0].id].data['train']) / len(dl_set[node_id].data['train']))
    load_node.get_child(1).prob = load_node.prob * (len(dl_set[child_nodes[1].id].data['train']) / len(dl_set[node_id].data['train']))

    get_labels_distribution(load_node, 'train')
    get_labels_distribution(load_node, 'test')

    nodes = {c_node.id: c_node for c_node in child_nodes}  # type: dict[int, GNode]

    nodes[child_nodes[0].id].set_trainer(dl_set[child_nodes[0].id], H, train_config, Model=GanImgTrainer)
    nodes[child_nodes[1].id].set_trainer(dl_set[child_nodes[1].id], H, train_config, Model=GanImgTrainer)

    load_node.get_child(0).gan.load_params(None, None, path = '../experiments/GANTree_MNIST_0to4_trial1/weights/iter/node'+str(child_nodes[0].id)+'/iter-40000.pt')
    load_node.get_child(1).gan.load_params(None, None, path = '../experiments/GANTree_MNIST_0to4_trial1/weights/iter/node'+str(child_nodes[1].id)+'/iter-40000.pt')    

    leaf_nodes.remove(node_id)
    leaf_nodes.update(load_node.child_ids)

    print('finished node-'+ str(node_id) + " loading")


## get new class to add to tree

new_class_dl = DataLoaderFactory.get_dataloader(H.dataloader, H.img_size, H.batch_size, H.batch_size, H.new_classes)
inc_threshold = 9.0
threshold = H.threshold
phase1_epochs = H.phase1_epochs
phase2_iters = H.phase2_iters


## get branch which is to be updated
incremental_nodes_branch = get_branch_incremental(new_class_dl, inc_threshold)
print(incremental_nodes_branch)


## update the branch

for l in range(len(incremental_nodes_branch)-1):

    train_inc_node_id = incremental_nodes_branch[l]

    train_inc_node = tree.nodes[train_inc_node_id]

    total_inc_train_examples = len(new_class_dl.data['train'])*(len(get_leaf_nodes(train_inc_node)))
    total_inc_test_examples = len(new_class_dl.data['test'])*(len(get_leaf_nodes(train_inc_node)))
    
    ## get new dataloader for the node
    train_dl_inc_new = get_new_inc_dataloader(train_inc_node, new_class_dl, total_inc_train_examples, total_inc_test_examples, incremental_nodes_branch)
    dl_set[train_inc_node_id] = train_dl_inc_new

    x_seed, l_seed = train_dl_inc_new.random_batch('test', 512)

    print(np.sum(train_inc_node.kmeans.pred == 0.))
    print(np.sum(train_inc_node.kmeans.pred == 1.))
    print(np.sum(train_inc_node.kmeans.pred == 2.))

    ## set new trainer for the node
    train_inc_node.set_trainer(train_dl_inc_new, H, train_config, Model=GanImgTrainer)

    if train_inc_node.get_child(0).id not in incremental_nodes_branch:
        for param in train_inc_node.get_child(0).gan.generator.parameters():
            param.requires_grad = False   

        for param in train_inc_node.get_child(0).gan.discriminator.parameters():
            param.requires_grad = False

    elif train_inc_node.get_child(1).id not in incremental_nodes_branch:
        for param in train_inc_node.get_child(1).gan.generator.parameters():
            param.requires_grad = False   

        for param in train_inc_node.get_child(1).gan.discriminator.parameters():
            param.requires_grad = False

    # if l == 0:
    #     train_inc_node = GNode.load('../experiments/GANTree_MNIST_0to4_trial1/weights/best_node_2_mnist_phase_1_2_inc_[5].pickle', train_inc_node)
    #     train_inc_node.child_nodes[train_inc_node.child_ids[0]] = GNode.load('../experiments/mnist_gan_tree_full_add5_threshold9/weights/best_node_0_child_0_mnist_phase_1_2_inc_[5].pickle', train_inc_node.child_nodes[train_inc_node.child_ids[0]])
    #     train_inc_node.child_nodes[train_inc_node.child_ids[1]] = GNode.load('../experiments/mnist_gan_tree_full_add5_threshold9/weights/best_node_0_child_1_mnist_phase_1_2_inc_[5].pickle', train_inc_node.child_nodes[train_inc_node.child_ids[1]])
    # elif l == 2:
    #     train_inc_node = GNode.load('../experiments/GANTree_MNIST_0to4_trial1/weights/best_node_2_mnist_phase_1_2_inc_[5].pickle', train_inc_node)
    #     train_inc_node.child_nodes[train_inc_node.child_ids[0]] = GNode.load('../experiments/mnist_gan_tree_full_add5_threshold9_trial2/weights/best_node_2_child_0_mnist_phase_1_2_inc_[5].pickle', train_inc_node.child_nodes[train_inc_node.child_ids[0]])
    #     train_inc_node.child_nodes[train_inc_node.child_ids[1]] = GNode.load('../experiments/mnist_gan_tree_full_add5_threshold9_trial2/weights/best_node_2_child_1_mnist_phase_1_2_inc_[5].pickle', train_inc_node.child_nodes[train_inc_node.child_ids[1]])

    train_phase_1(train_inc_node, threshold = threshold, phase1_epochs = phase1_epochs)
    if H.save_node:
        save_nodes(train_inc_node, 'phase_1_2_inc_' + str(H.new_classes))

    split_dataloader(train_inc_node)

    train_inc_node.get_child(0).set_trainer(dl_set[train_inc_node.get_child(0).id], H, train_config, Model=GanImgTrainer)
    train_inc_node.get_child(1).set_trainer(dl_set[train_inc_node.get_child(1).id], H, train_config, Model=GanImgTrainer)

    ## train the node with the new dataloader, normal 1 output discriminator
    # if l == 0:
    #     train_inc_node.get_child(0).gan.load_params(None, None, path = '../experiments/mnist_gan_tree_full_add5_threshold9/weights/iter/node1/iter-20000.pt')
    #     train_inc_node.get_child(1).gan.load_params(None, None, path = '../experiments/mnist_gan_tree_full_add5_threshold9/weights/iter/node2/iter-20000.pt')    
    # elif l == 2:
    #     train_inc_node.get_child(0).gan.load_params(None, None, path = '../experiments/mnist_gan_tree_full_add5_threshold9_trial2/weights/iter/node5/iter-15000.pt')
    #     train_inc_node.get_child(1).gan.load_params(None, None, path = '../experiments/mnist_gan_tree_full_add5_threshold9_trial2/weights/iter/node6/iter-15000.pt')    

    train_phase_2(train_inc_node, phase2_iters, incremental_nodes_branch)
    if H.save_node:
        save_nodes(train_inc_node, 'phase_2_inc_' + str(H.new_classes))


## new parent adding starts
print("new parent adding starts")

if len(incremental_nodes_branch) == 1:
    base_id = len(tree.nodes)
    new_parent_node_id = base_id + 1
    new_parent_model_name = "node%d" % new_parent_node_id

    new_parent_model = ImgGAN.create_from_hyperparams(new_parent_model_name, H, cov_sign = '0')
    means = as_np(gan.z_op_params.means)
    cov = as_np(gan.z_op_params.cov)
    dist_params = DistParams(means=means, cov=cov, pi=1.0, prob=1.0)
    child_number = 0
    n_dim = means.shape[0]

    new_parent_node = tree.create_child_node(dist_params, new_parent_model)

else:
    parent_node = tree.nodes[incremental_nodes_branch[-2]]
    child_number = 0 if parent_node.child_ids[0] in incremental_nodes_branch else 1
    means = parent_node.kmeans.means[child_number]
    cov = parent_node.kmeans.covs[child_number]
    dist_params = DistParams(means=means, cov=cov, pi=parent_node.kmeans.weights[child_number], prob=parent_node.get_child(child_number).prob)
    Model = parent_node.gan.__class__
    n_dim = means.shape[0]

    new_parent_common_encoder = parent_node.gan.encoder.copy(n_dim, parent_node.gan.channel)


    base_id = len(tree.nodes)
    new_parent_node_id = base_id + 1
    new_parent_model_name = "node%d" % new_parent_node_id

    new_parent_model = Model(name = new_parent_model_name,
                        z_op_params = (tr.Tensor(means), tr.Tensor(cov)),
                        encoder = new_parent_common_encoder,
                        generator = parent_node.get_child(child_number).gan.generator.copy(n_dim, parent_node.gan.channel),
                        discriminator = parent_node.get_child(child_number).gan.discriminator.copy(n_dim)
                        )

    new_parent_node = tree.create_child_node(dist_params, new_parent_model, parent_node)

dmu = H.dmu
value = 0.5 * dmu / math.sqrt(H.z_dim)

means1 = [new_parent_node.prior_means[i] + value for i in range(H.z_dim)]
means2 = [new_parent_node.prior_means[i] - value for i in range(H.z_dim)]
means = np.asarray([means1, means2])

covs1 = np.eye(H.z_dim)
covs2 = np.eye(H.z_dim)
covs = np.asarray([covs1, covs2])


new_parent_node.kmeans = KMeansCltr(means, covs, None, None, None, None)

## Create new child node

new_child_means = new_parent_node.kmeans.means[1 - child_number]
new_child_cov = new_parent_node.kmeans.covs[1 - child_number]
new_child_dist_params = DistParams(means=new_child_means, cov=new_child_cov, pi=1.0, prob = 1.0 / (len(get_leaf_nodes(tree.nodes[incremental_nodes_branch[-1]]))+1))
Model = new_parent_node.gan.__class__

new_child_common_encoder = new_parent_node.gan.encoder.copy(n_dim, new_parent_node.gan.channel)

base_id = len(tree.nodes)
new_child_node_id = base_id + 1
new_child_model_name = "node%d" % new_child_node_id

new_child_model = Model(name = new_child_model_name,
                        z_op_params = (tr.Tensor(new_child_means), tr.Tensor(new_child_cov)),
                        encoder = new_child_common_encoder,
                        generator = new_parent_node.gan.generator.copy(n_dim, new_parent_node.gan.channel),
                        discriminator = new_parent_node.gan.discriminator.copy(n_dim)
                        )

new_child_node = tree.create_child_node(new_child_dist_params, new_child_model, new_parent_node)


## update assignment of parent and child

new_parent_node.child_ids = [0, 0]


tree.nodes[incremental_nodes_branch[-1]].parent = new_parent_node
new_parent_node.child_ids[1-child_number] = new_child_node.id
new_parent_node.child_ids[child_number] = incremental_nodes_branch[-1]

new_parent_node.child_nodes[new_child_node.id] = new_child_node
new_parent_node.child_nodes[incremental_nodes_branch[-1]] = tree.nodes[incremental_nodes_branch[-1]]

new_parent_node.set_optimizer()

## set trainer of new parent node and train if new parent node is root

total_inc_train_examples = len(new_class_dl.data['train'])*(len(get_leaf_nodes(tree.nodes[incremental_nodes_branch[-1]])))
total_inc_test_examples = len(new_class_dl.data['test'])*(len(get_leaf_nodes(tree.nodes[incremental_nodes_branch[-1]])))

dl_new_parent = generate_new_parent_dl(new_parent_node, new_class_dl, child_number, total_inc_train_examples, total_inc_test_examples)
x_seed, l_seed = dl_new_parent.random_batch('test', 512)
new_parent_node.set_trainer(dl_new_parent, H, train_config, Model=GanImgTrainer)

if len(incremental_nodes_branch) == 1:
    new_parent_node.train(H.root_gan_iters)


# new_parent_node.init_child_params(dl_new_parent.data['train'], n_components = 2, applyPCA = False, H = H)
new_parent_node.kmeans.pred = dl_new_parent.labels['train'].numpy()


## train child node

train_phase_1(new_parent_node, threshold = threshold, phase1_epochs = phase1_epochs)
if H.save_node:
    save_nodes(new_parent_node, 'phase_1_2_add_last')
# new_parent_node = GNode.load('../experiments/mnist_gan_tree_full_add5_threshold9_total/weights/best_node_7_mnist_phase_1_2_add_last.pickle', new_parent_node)
# new_parent_node.child_nodes[new_parent_node.child_ids[0]] = GNode.load('../experiments/mnist_gan_tree_full_add5_threshold9_total/weights/best_node_7_child_0_mnist_phase_1_2_add_last.pickle', new_parent_node.child_nodes[new_parent_node.child_ids[0]])
# new_parent_node.child_nodes[new_parent_node.child_ids[1]] = GNode.load('../experiments/mnist_gan_tree_full_add5_threshold9_total/weights/best_node_7_child_1_mnist_phase_1_2_add_last.pickle', new_parent_node.child_nodes[new_parent_node.child_ids[1]])

split_dataloader(new_parent_node)

new_parent_node.child_nodes[new_parent_node.child_ids[0]].set_trainer(dl_set[new_parent_node.child_ids[0]], H, train_config, Model=GanImgTrainer)
new_parent_node.child_nodes[new_parent_node.child_ids[1]].set_trainer(dl_set[new_parent_node.child_ids[1]], H, train_config, Model=GanImgTrainer)

train_phase_2(new_parent_node, phase2_iters, [new_child_node.id, incremental_nodes_branch[-1]])
if H.save_node:
    save_nodes(new_parent_node, 'phase_2_add_last')

print('finished incremental training')
