from __future__ import print_function, division
import os, argparse, logging, json

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys, time
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult

from tqdm import tqdm
import numpy as np

import torch as tr
from matplotlib import pyplot as plt
from configs import Config

default_args_str = '-hp hyperparams/mixed_mnist.py -en exp20_mixed_mnist'

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
    n_step_tboard_log=25,
    n_step_console_log=-1,
    n_step_validation=100,
    n_step_save_params=2500,
    n_step_visualize=500
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


# nodes[1].update_dist_params(means=2 * np.ones(2), cov=np.eye(2), prior_prob=1)
# nodes[2].update_dist_params(means=- 2 * np.ones(2), cov=np.eye(2), prior_prob=1)

def get_x_clf_plot_data(root, x_batch):
    with tr.no_grad():
        z_batch_post = root.post_gmm_encode(x_batch)
        x_recon_post, _ = root.post_gmm_decode(z_batch_post)

        z_batch_pre = root.pre_gmm_encode(x_batch)
        x_recon_pre = root.pre_gmm_decode(z_batch_pre)

        z_batch_post = as_np(z_batch_post)
        x_recon_post = as_np(x_recon_post)

        z_batch_pre = as_np(z_batch_pre)
        x_recon_pre = as_np(x_recon_pre)

    return z_batch_pre, z_batch_post, x_recon_pre, x_recon_post


def get_deep_type(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [get_deep_type(o) for o in obj]
    return str(type(obj))


def get_plot_data(iter_no, node, x_batch, labels):
    # type: (int, GNode, np.ndarray, np.ndarray) -> list
    z_batch_pre, z_batch_post, x_recon_pre, x_recon_post = get_x_clf_plot_data(node, x_batch)

    z_rand0 = node.get_child(0).sample_z_batch(x_batch.shape[0])
    z_rand1 = node.get_child(1).sample_z_batch(x_batch.shape[0])
    with tr.no_grad():
        x_fake0 = node.post_gmm_decoders[0].forward(z_rand0)
        x_fake1 = node.post_gmm_decoders[1].forward(z_rand1)

    plot_data = [
        [
            [
                as_np(x_batch),
                as_np(labels).astype(int)
            ],
            as_np(node.dist_params),
            node.get_child(0).dist_params,
            node.get_child(1).dist_params
        ], [
            z_batch_pre,
            z_batch_post,
            x_recon_pre,
            x_recon_post
        ], [
            as_np(z_rand0),
            as_np(x_fake0),
            as_np(z_rand1),
            as_np(x_fake1),
        ]
    ]
    return plot_data


def generate_plots(plot_data, iter_no, tag):
    fig = get_x_clf_figure(plot_data)
    path = Paths.get_result_path('%s_%03d' % (tag, iter_no))
    fig.savefig(path)
    plt.close(fig)
    return (iter_no, path)


def visualize_plots(iter_no, node, x_batch, labels, tag):
    plot_data = get_plot_data(iter_no, node, x_batch, labels)
    future = pool.apply_async(generate_plots, (plot_data, iter_no, tag))
    future_objects.append(future)
    return future


def train_phase_1(node, n_iterations):
    node.fit_gmm(x_seed)
    visualize_plots(iter_no=0, node=node, x_batch=x_seed, labels=l_seed, tag='x_clf_plots')

    with tqdm(total=n_iterations) as pbar:
        for iter_no in range(n_iterations):
            node.trainer.iter_no = iter_no

            i = iter_no + 1

            # Training common encoder over cross-classification loss with a batch across common dataloader
            x_clf_train_batch, _ = dl_set[0].next_batch('train')
            z_batch, x_recon, x_recon_loss, x_clf_loss, loss = node.step_train_x_clf(x_clf_train_batch)

            node.trainer.writer['train'].add_scalar('x_clf_loss', x_clf_loss, iter_no)
            node.trainer.writer['train'].add_scalar('x_recon_loss', x_recon_loss, iter_no)
            node.trainer.writer['train'].add_scalar('loss', loss, iter_no)

            node.fit_gmm(x_seed)
            if i < 10 or i % 10 == 0:
                visualize_plots(iter_no=i, node=node, x_batch=x_seed, labels=l_seed, tag='x_clf_plots')
            pbar.update(n=1)


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
            if is_gan_vis_iter(iter_no):
                visualize_plots(iter_no, root, x_seed, l_seed, tag='gan_plots')


def train_node(node, x_clf_iters=200, gan_iters=10000):
    # type: (GNode, int, int) -> None
    global future_objects
    child_nodes = tree.split_node(node, fixed=False)

    nodes = {node.id: node for node in child_nodes}  # type: dict[int, GNode]

    relabel_samples(node)

    nodes[1].set_trainer(dl_set[1], H, train_config)
    nodes[2].set_trainer(dl_set[2], H, train_config)

    future_objects = []  # type: list[ApplyResult]

    train_phase_1(node, x_clf_iters)

    relabel_samples(node)

    train_phase_2(node, gan_iters)
    # Logging the image savingh operations status
    for i, obj in enumerate(future_objects):
        iter_no, path = obj.get()
        if obj.successful():
            print('Saved figure for iter %3d @ %s' % (iter_no, path))
        else:
            print('Failed saving figure for iter %d' % iter_no)


gan = ImgGAN.create_from_hyperparams('node0', H, '0')
means = as_np(gan.z_op_params.means)
cov = as_np(gan.z_op_params.cov)
dist_params = DistParams(means=means, cov=cov, pi=1.0, prob=1.0)

dl = DataLoaderFactory.get_dataloader(H.dataloader, H.batch_size, H.batch_size)

x_seed, l_seed = dl.random_batch('test', 32)

tree = GanTree('gtree', ImgGAN, H, x_seed)
root = tree.create_child_node(dist_params, gan)

root.set_trainer(dl, H, train_config, Model=GanImgTrainer)

# # # GNode.load('best_node.pickle', root)
# for i in range(20):
#     root.train(5000)
#     root.save('../experiments/' + exp_name + '/best_node-' + str(i) + '.pt')

# dl_set = {0: dl}
#
# future_objects = []  # type: list[ApplyResult]
#
# pool = Pool(processes=16)
#
# root.save('best_root_phase1.pickle')
# root.get_child(0).save('best_child0_phase1.pickle')
# root.get_child(1).save('best_child1_phase1.pickle')
# # print('Iter: %d' % (iter_no + 1))
# print('Training Complete.')
