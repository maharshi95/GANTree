from __future__ import print_function, division
import os, argparse, logging, json

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import pickle
import sys, time
import numpy as np
import torch as tr
import matplotlib
from termcolor import colored

matplotlib.use('Agg')

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult
from matplotlib import pyplot as plt

from configs import Config
from exp_context import ExperimentContext

default_args_str = '-hp base/hyperparams.py -d all -en exp22_toy_gantree_disc_z -g 1'

if Config.use_gpu:
    print('mode: GPU')
    tr.set_default_tensor_type('torch.cuda.FloatTensor')

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
ExperimentContext.set_context(args.hyperparams, args.exp_name)
H = ExperimentContext.Hyperparams  # type: Hyperparams
exp_name = ExperimentContext.exp_name

##########  Set Logging  ###########
logger = logging.getLogger(__name__)
LOG_FORMAT = "[{}: %(filename)s: %(lineno)3s] %(levelname)s: %(funcName)s(): %(message)s".format(ExperimentContext.exp_name)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

gpu_idx = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

#### Clear Logs and Results based on the argument flags ####
from paths import Paths
from utils.tr_utils import as_np
from utils import bash_utils, model_utils, viz_utils

if 'all' in args.delete or 'logs' in args.delete or resume_flag is False:
    logger.warning('Deleting Logs...')
    bash_utils.delete_recursive(Paths.logs_base_dir)
    print('')

if 'all' in args.delete or 'results' in args.delete:
    logger.warning('Deleting all results in {}...'.format(Paths.results_base_dir))
    bash_utils.delete_recursive(Paths.results_base_dir)
    print('')

##### Create required directories
logger.info('Created directories')
model_utils.setup_dirs(log_flag=False)

##### Model and Training related imports
from dataloaders.factory import DataLoaderFactory
from dataloaders.custom_loader import CustomDataLoader
from base.hyperparams import Hyperparams

from models.toy.gan import ToyGAN
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

    logger.info(colored(tensorboard_msg, 'blue', attrs=['bold']))
    # display(HTML(html_content))

# Dump Hyperparams file the experiments directory
hyperparams_string_content = json.dumps(H.__dict__, default=lambda x: repr(x), indent=4, sort_keys=True)
# print(hyperparams_string_content)
with open(Paths.exp_hyperparams_file, "w") as fp:
    fp.write(hyperparams_string_content)

train_config = TrainConfig(
    n_step_tboard_log=5,
    n_step_console_log=-1,
    n_step_validation=100,
    n_step_save_params=2000,
    n_step_visualize=100
)


def full_train_step(gnode, dl, visualize=True, validation=True, save_params=True):
    """
    :type gnode: GNode
    """
    trainer = gnode.trainer
    model = gnode.gan  # type: ToyGAN
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

    # # Console Log
    # if trainer.is_console_log_step():
    #     print('============================================================')
    #     print('Train Step', trainer.iter_no + 1)
    #     print('%s: step %i:     Disc Acc: %.3f' % (exp_name, trainer.iter_no, metrics['accuracy_dis_x'].item()))
    #     print('%s: step %i:     Gen  Acc: %.3f' % (exp_name, trainer.iter_no, metrics['accuracy_gen_x'].item()))
    #     print('%s: step %i: x_recon Loss: %.3f' % (exp_name, trainer.iter_no, metrics['loss_x_recon'].item()))
    #     print('%s: step %i: z_recon Loss: %.3f' % (exp_name, trainer.iter_no, metrics['loss_z_recon'].item()))
    #     print('------------------------------------------------------------')

    # Tensorboard Log
    if trainer.is_tboard_log_step():
        for tag, value in metrics.items():
            trainer.writer['train'].add_scalar(tag, value.item(), trainer.iter_no)

    # # Validation Computations
    # if validation and trainer.is_validation_step():
    #     trainer.validation()

    # # Weights Saving
    # if save_params and trainer.is_params_save_step():
    #     tic_save = time.time()
    #     model.save_params(dir_name='iter', weight_label='iter', iter_no=trainer.iter_no)
    #     tac_save = time.time()
    #     save_time = tac_save - tic_save
    #     if trainer.is_console_log_step():
    #         print('Param Save Time: %.4f' % (save_time))
    #         print('------------------------------------------------------------')

    # # Visualization
    # if visualize and trainer.is_visualization_step():
    #     # previous_backend = plt.get_backend()
    #     # plt.switch_backend('Agg')
    #     trainer.visualize('train')
    #     trainer.visualize('test')
    #     # plt.switch_backend(previous_backend)

    # Switch Training Networks - Gen | Disc
    trainer.switch_train_mode(g_acc, d_acc)

    # iter_time_end = time.time()
    # if trainer.is_console_log_step():
    #     print('Total Iter Time: %.4f' % (iter_time_end - iter_time_start))
    #     if trainer.tensorboard_msg:
    #         print('------------------------------------------------------------')
    #         print(trainer.tensorboard_msg)
    #     print('============================================================')
    #     print()


def split_dataloader(node):
    dl = dl_set[node.id]

    train_splits, train_split_index = node.split_x(dl.data['train'])
    test_splits, test_split_index = node.split_x(dl.data['test'])

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
        dl_set[i] = CustomDataLoader.create_from_parent(dl, data_tuples[:index])
        seed_data[i] = dl_set[i].random_batch('train', 2048)
    with open('seed_data.pickle', 'w') as fp:
        pickle.dump(seed_data, fp)


def get_x_clf_plot_data(node, x_batch):
    with tr.no_grad():
        z_batch_post = node.post_gmm_encode(x_batch)
        x_recon_post, _ = node.post_gmm_decode(z_batch_post)

        z_batch_pre = node.pre_gmm_encode(x_batch)
        x_recon_pre = node.pre_gmm_decode(z_batch_pre)

        z_batch_post = as_np(z_batch_post)
        x_recon_post = as_np(x_recon_post)

        z_batch_pre = as_np(z_batch_pre)
        x_recon_pre = as_np(x_recon_pre)

    return z_batch_pre, z_batch_post, x_recon_pre, x_recon_post


def get_deep_type(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [get_deep_type(o) for o in obj]
    return str(type(obj))


def get_plot_data(node, x_batch, labels):
    # type: (GNode, np.ndarray, np.ndarray) -> list
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


def generate_plots(plot_data, iter_no, tag, model_name=None):
    fig = viz_utils.get_x_clf_figure(plot_data)
    path = Paths.get_result_path('%s_%05d' % (tag, iter_no), model_name)
    fig.savefig(path)
    plt.close(fig)
    return (iter_no, path)


def visualize_plots(iter_no, node, x_batch, labels, tag):
    plot_data = get_plot_data(node, x_batch, labels)
    # generate_plots(plot_data, iter_no, tag)
    future = pool.apply_async(generate_plots, (plot_data, iter_no, tag, node.name))
    future_objects.append(future)
    return future


def save_node(node, tag=None, iter=None):
    # type: (GNode, str, int) -> None
    filename = node.name
    if tag is not None:
        filename += str(tag)
    if iter is not None:
        filename += ('%05d' % iter)
    filename = filename + '.pt'
    filepath = os.path.join(Paths.weight_dir_path(''), filename)
    node.save(filepath)


def load_node(node_name, tag):
    filename = '%s_%s.pt' % (node_name, tag) if tag else node_name + '.pt'
    filepath = os.path.join(Paths.weight_dir_path(''), filename)
    gnode = GNode.load(filepath, Model=ToyGAN)
    return gnode


def train_phase_1(node, n_iterations):
    x_seed, l_seed = seed_data[node.id]

    logger.info(colored('Training X-CLF (Phase 1) on %s' % node.name, 'yellow', attrs=['bold']))

    node.fit_gmm(x_seed)
    visualize_plots(iter_no=0, node=node, x_batch=x_seed, labels=l_seed, tag='phase1_plot')

    with tqdm(total=n_iterations) as pbar:
        for iter_no in range(n_iterations):
            node.trainer.iter_no = iter_no

            i = iter_no + 1

            # Training common encoder over cross-classification loss with a batch across common dataloader
            x_clf_train_batch, _ = dl_set[node_id].next_batch('train')
            z_batch, x_recon, x_recon_loss, x_clf_loss, loss = node.step_train_x_clf(x_clf_train_batch)

            node.trainer.writer['train'].add_scalar('x_clf_loss', x_clf_loss, iter_no)
            node.trainer.writer['train'].add_scalar('x_recon_loss', x_recon_loss, iter_no)
            node.trainer.writer['train'].add_scalar('loss', loss, iter_no)

            node.fit_gmm(x_seed)
            if i < 10 or i < 500 and i % 10 == 0 or i % 100 == 0:
                visualize_plots(iter_no=i, node=node, x_batch=x_seed, labels=l_seed, tag='phase1_plot')
                save_node(node, 'half', iter_no)
            pbar.update(n=1)


def is_gan_vis_iter(i):
    return ((i <= 1000 and i % 50 == 0)
            or (1000 < i <= 5000 and i % 100 == 0)
            or (5000 < i <= 10000 and i % 200 == 0))


def train_phase_2(node, n_iterations, min_iters, x_clf_limit, x_recon_limit):
    # type: (GNode, int, int, float, float) -> None
    logger.info(colored('Training I-GAN (Phase 2) on %s' % node.name, 'yellow', attrs=['bold']))

    x_seed, l_seed = seed_data[node.id]
    with tqdm(total=n_iterations) as pbar:
        for iter_no in range(n_iterations):
            for i in node.child_ids:
                full_train_step(node.child_nodes[i], dl_set[i], visualize=False)
                pbar.update(n=0.5)

            x_batch_left, _ = dl_set[node.left.id].next_batch('train')
            x_batch_right, _ = dl_set[node.right.id].next_batch('train')

            x_recon_loss, x_clf_loss, loss = node.step_train_x_clf_fixed(x_batch_left, x_batch_right, clip=x_clf_limit)
            node.trainer.writer['train'].add_scalar('x_clf_loss_post', x_clf_loss, iter_no)

            if is_gan_vis_iter(iter_no):
                visualize_plots(iter_no, node, x_seed, l_seed, tag='phase2_plot')
                save_node(node, 'full', iter_no)

            if iter_no >= min_iters and x_clf_loss <= x_clf_limit * (1.0001) and x_recon_loss <= x_recon_limit * (1.0001):
                break


def train_node(node, x_clf_iters, gan_iters, min_gan_iters, x_clf_lim, x_recon_limit):
    # type: (GNode, int, int, int ,float, float) -> None
    global future_objects
    child_nodes = tree.split_node(node, fixed=False)

    node.fit_gmm(seed_data[node.id][0], warm_start=False)
    split_dataloader(node)
    for cnode in child_nodes:
        cnode.set_trainer(dl_set[cnode.id], H, train_config)

    future_objects = []  # type: list[ApplyResult]

    bash_utils.create_dir(Paths.get_result_path('', node.name), log_flag=False)
    train_phase_1(node, x_clf_iters)

    # Saving Node
    save_node(node, 'full_fit')
    for cid, cnode in node.child_nodes.items():
        save_node(cnode, 'half')

    # Splitting Dataloader
    split_dataloader(node)

    train_phase_2(node, gan_iters, min_iters=min_gan_iters, x_clf_limit=x_clf_lim, x_recon_limit=x_recon_limit)

    split_dataloader(node)

    for cid, cnode in node.child_nodes.items():
        save_node(cnode, 'full')

    # Logging the image savingh operations status
    n_failed = 0
    path = ''
    for i, obj in enumerate(future_objects):
        iter_no, path = obj.get()
        if not obj.successful():
            n_failed += 1
    if n_failed > 0:
        print('Attempted saving %d images at %s, Failed: %s' % (len(future_objects), os.path.dirname(path), n_failed))
    else:
        print('%d images successfully saved at %s' % (len(future_objects) + n_failed, os.path.dirname(path)))


def find_next_node():
    logger.info(colored('Leaf Nodes: %s' % str(list(leaf_nodes)), 'green', attrs=['bold']))
    likelihoods = {i: tree.nodes[i].mean_likelihood(dl_set[i].data['train']) for i in leaf_nodes}
    n_samples = {i: dl_set[i].data['train'].shape[0] for i in leaf_nodes}
    pairs = [(node_id, n_samples[node_id], likelihoods[node_id]) for node_id in leaf_nodes]
    for pair in pairs:
        logger.info('Node: %2d N_Samples: %5d Likelihood %.03f' % (pair[0], pair[1], pair[2]))
    min_samples = min(n_samples)
    for leaf_id in leaf_nodes:
        if n_samples[leaf_id] > 3 * min_samples:
            return max(leaf_nodes, key=lambda i: n_samples[i])
    return min(leaf_nodes, key=lambda i: likelihoods[i])


gan = ToyGAN.create_from_hyperparams('node0', H, '0')
means = as_np(gan.z_op_params.means)
cov = as_np(gan.z_op_params.cov)
dist_params = DistParams(means=means, cov=cov, pi=1.0, prob=1.0)

dl = DataLoaderFactory.get_dataloader(H.dataloader, H.input_size, H.z_size, H.batch_size, H.batch_size, supervised=True)
x_seed, l_seed = dl.random_batch('test', 2048)
tree = GanTree('gtree', ToyGAN, H, x_seed)
root = tree.create_child_node(dist_params, gan)

root.set_trainer(dl, H, train_config)

# GNode.load('9g_root.pickle', root)
logger.info(colored('Training Root Node for GAN Training', 'green', attrs=['bold']))
root.train(20000)
root.save('toy_root_disc_z.pickle')

dl_set = {0: dl}

seed_data = {
    0: (x_seed, l_seed)
}

leaf_nodes = {0}

future_objects = []  # type: list[ApplyResult]

pool = Pool(processes=16)

bash_utils.create_dir(Paths.weight_dir_path(''), log_flag=False)
for i_modes in range(8):
    node_id = find_next_node()
    logger.info(colored('Next Node to split: %d' % node_id, 'green', attrs=['bold']))
    node = tree.nodes[node_id]
    train_node(node, x_clf_iters=1000, gan_iters=20000, min_gan_iters=5000, x_clf_lim=0.00001, x_recon_limit=0.004)
    leaf_nodes.remove(node_id)
    leaf_nodes.update(node.child_ids)
    print('')

# root.save('best_root_phase1.pickle')
# root.get_child(0).save('best_child0_phase1.pickle')
# root.get_child(1).save('best_child1_phase1.pickle')
# # print('Iter: %d' % (iter_no + 1))
# print('Training Complete.')
