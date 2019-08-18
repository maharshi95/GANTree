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

from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt


# default_args_str = '-t'

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
parser.add_argument('-en', '--exp_name', required=True, help='experiment name')

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


##### Model and Training related imports
from dataloaders.factory import DataLoaderFactory
from base.hyperparams import Hyperparams
from trainers.gan_image_trainer import GanImgTrainer
from trainers.gan_image_trainer import save_image
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
        elif split == 'test':
            data = x_seed

        Z = node.post_gmm_encode(data)

        labels = node.gmm_predict_test(Z, threshold).tolist()

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
        elif split == 'test':
            data = x_seed

        preds = node.gmm_predict(node.post_gmm_encode(data))
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

        node.trainer.writer[split].add_image(node.name + '_' + phase + '_child0_recon', recon_img_child0, iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child0_gen', gen_img_child0, iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child0_real', real_img_child0, iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child1_recon', recon_img_child1, iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child1_gen', gen_img_child1, iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child1_real', real_img_child1, iter_no)


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
            p = node.gmm_predict_test(Z, threshold)

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
        # plt.hist(plot_data_tensorboard, bins = 'auto', color = ['g'])

        fig_mean_axis_histogram = plt.gcf()
        node.trainer.writer[split].add_histogram(node.name + '_' + phase + '_mean_axis_' + str(i), plot_data_tensorboard, iter_no)
        # node.trainer.writer[split].add_image(node.name + '_mean_axis_' + str(i), fig_mean_axis_histogram, iter_no)
        path_mean_axis_hist = Paths.get_result_path(node.name + '_' + split +  '_mean_axis_histogram/' + phase + '%03d_%01d' % (iter_no, i))
        fig_mean_axis_histogram.savefig(path_mean_axis_hist)
        plt.close(fig_mean_axis_histogram)



def train_phase_1(node, threshold = 2.5, phase1_epochs = 7):
    print('entered phase 1')

    print(distance.mahalanobis(node.kmeans.means[1], node.kmeans.means[0], node.kmeans.covs[0]))
    print(threshold)

    reassignment_epoch = 0


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

        if j == reassignment_epoch:
            node.reassignLabels(train_data, threshold)


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


def train_phase_2(node, n_iterations):
    node.get_child(0).trainer.iter_no = 0 
    node.get_child(1).trainer.iter_no = 0    

    # type: (GNode, int) -> None
    with tqdm(total=n_iterations) as pbar:
        for iter_no in range(n_iterations):
            for i in node.child_ids:
                node.child_nodes[i].trainer.train(1, enable_tqdm = False)
                pbar.update(n=0.5)

    z_histogram_plot(node.get_child(0), 'train', iter_no, 'phase_2')
    z_histogram_plot(node.get_child(0), 'test', iter_no, 'phase_2')

    z_histogram_plot(node.get_child(1), 'train', iter_no, 'phase_2')
    z_histogram_plot(node.get_child(1), 'test', iter_no, 'phase_2')


def train_node(node, phase1_epochs = 2, phase2_iters = 10000):
    # type: (GNode, int, int) -> None
    threshold = H.threshold

    child_nodes = tree.split_node(node, x_batch = dl_set[node.id].data['train'], applyPCA = False, H = H)
    # child_nodes = tree.load_children(node, '../experiments/mnist_node1/weights/best_node_' + str(node.id) + '_child_')

    train_phase_1(node, threshold = threshold, phase1_epochs = phase1_epochs)
    if H.save_node:
        save_nodes(node, 'phase_1_2')

    split_dataloader(node)

    get_labels_distribution(node, 'train')
    get_labels_distribution(node, 'test')

    nodes = {node.id: node for node in child_nodes}  # type: dict[int, GNode]

    nodes[child_nodes[0].id].set_trainer(dl_set[child_nodes[0].id], H, train_config, Model=GanImgTrainer)
    nodes[child_nodes[1].id].set_trainer(dl_set[child_nodes[1].id], H, train_config, Model=GanImgTrainer)

    train_phase_2(node, phase2_iters)
    if H.save_node:
        save_nodes(node, 'phase_2')

    # node.get_child(0).gan.load_params(None, None, path = '../experiments/mnist_root_trial11/weights/iter/node1/iter-10000.pt')
    # node.get_child(1).gan.load_params(None, None, path = '../experiments/mnist_root_trial11/weights/iter/node2/iter-10000.pt')    


def likelihood(node, dl):
    samples = dl.data['train'].shape[0]
    X_complete = dl.data['train']

    iter = samples // 256

    p = np.zeros([iter], dtype=np.float32)

    for idx in range(iter):
        p[idx] = node.mean_likelihood(X_complete[(idx) * 256:(idx + 1) * 256].cuda())

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


def print_gpu_stats():
    print(tr.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(tr.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(tr.cuda.memory_cached(0)/1024**3,1), 'GB')


def save_nodes(node, phase):
    node.save('../experiments/'+ exp_name + '/weights/best_node_' + str(node.id) + '_mnist_' + str(phase)+'.pickle')
    node.child_nodes[node.child_ids[0]].save('../experiments/'+ exp_name + '/weights/best_node_' + str(node.id) + '_child_0_mnist_' + str(phase) + '.pickle')
    node.child_nodes[node.child_ids[1]].save('../experiments/'+ exp_name + '/weights/best_node_' + str(node.id) + '_child_1_mnist_' + str(phase) + '.pickle')



gan = ImgGAN.create_from_hyperparams('node0', H, cov_sign = '0')
means = as_np(gan.z_op_params.means)
cov = as_np(gan.z_op_params.cov)
dist_params = DistParams(means=means, cov=cov, pi=1.0, prob=1.0)

dl = DataLoaderFactory.get_dataloader(H.dataloader, H.img_size, H.batch_size, H.batch_size)

x_seed, l_seed = dl.random_batch('test', 512)

tree = GanTree('gtree', ImgGAN, H, x_seed)
root = tree.create_child_node(dist_params, gan)
root.set_trainer(dl, H, train_config, Model=GanImgTrainer)

root.train(H.root_gan_iters)
# root.gan.load_params(None, None, path = '../experiments/GANTree_Fashion_MNIST_Mixed_trial1/weights/iter/node0/iter-80000.pt')
# root = GNode.load('../experiments/GANTree_Fashion_MNIST_Mixed_trial1/weights/best_node_0_mnist_phase_1_2.pickle', root)

dl_set = {0: dl}
leaf_nodes = {0}

bash_utils.create_dir(Paths.weight_dir_path(''), log_flag=False)

for i_modes in range(10):
    node_id = find_next_node()

    logger.info(colored('Next Node to split: %d' % node_id, 'green', attrs=['bold']))

    node = tree.nodes[node_id]

    x_seed, l_seed = dl_set[node_id].random_batch('test', 512)

    train_node(node, phase1_epochs = H.phase1_epochs, phase2_iters = H.phase2_iters)

    leaf_nodes.remove(node_id)
    leaf_nodes.update(node.child_ids)

    print('finished node-'+ str(node.id) + " training")
    