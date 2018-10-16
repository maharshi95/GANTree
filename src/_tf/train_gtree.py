from __future__ import division, print_function, absolute_import

__doc__ = "Archived training algorithm for gan tree"
import json
import os, argparse, logging, traceback

import numpy as np
import matplotlib

matplotlib.use('Agg')
from exp_context import ExperimentContext

# Setting up Argument parser

parser = argparse.ArgumentParser()

parser.add_argument('-g', '--gpu', default=0, help='index of the gpu to be used. default: 0')
parser.add_argument('-r', '--resume', nargs='?', const=True, default=False,
                    help='if present, the training resumes from the latest step, '
                         'for custom step number, provide it as argument value')
parser.add_argument('-d', '--delete', nargs='+', default=[], choices=['logs', 'weights', 'results', 'all'],
                    help='delete the entities')
parser.add_argument('-w', '--weights', nargs='?', default='iter', choices=['iter', 'best_gen', 'best_pred'],
                    help='weight type to load if resume flag is provided. default: iter')
parser.add_argument('-hp', '--hyperparams', required=True, help='hyperparam class to use from HyperparamFactory')
parser.add_argument('-en', '--exp_name', default=None, help='experiment name. if not provided, it is taken from Hyperparams')

args = parser.parse_args()

ExperimentContext.set_context(args.hyperparams, args.exp_name)

logger = logging.getLogger(__name__)
LOG_FORMAT = "[{}: %(filename)s: %(lineno)3s] %(levelname)s: %(funcName)s(): %(message)s".format(ExperimentContext.exp_name)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

from _tf import paths
from utils import np_utils
from utils import bash_utils, model_utils
from dataloaders.factory import DataLoaderFactory

from _tf.gan_tree import gan_tree

gpu_idx = str(args.gpu)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

resume_flag = args.resume is not False

if 'all' in args.delete or 'logs' in args.delete or resume_flag is False:
    logger.warning('Deleting Logs...')
    bash_utils.delete_recursive(paths.logs_base_dir)
    print('')

if 'all' in args.delete or 'results' in args.delete:
    logger.warning('Deleting all results in {}...'.format(paths.results_base_dir))
    bash_utils.delete_recursive(paths.results_base_dir)
    print('')

model_utils.setup_dirs()

Model = ExperimentContext.get_model_class()
H = ExperimentContext.get_hyperparams()
print('input_size:', H.input_size, 'latent_size:', H.z_size)
dl = DataLoaderFactory.get_dataloader(H.dataloader, H.input_size, H.z_size)

hyperparams_string_content = json.dumps(H.__dict__, default=lambda x: repr(x), indent=4, sort_keys=True)
print(hyperparams_string_content)
with open(paths.exp_hyperparams_file, "w") as fp:
    fp.write(hyperparams_string_content)

x_train, x_test = dl.get_data()
print('Train Test Data loaded...')

gtree = gan_tree.GanTree('gan-tree', Model, x_test)
gtree.initiate()
print('GAN Tree service initiated...Current active nodes: {}'.format(gtree.n_active_nodes))

## TODO: Decide the mechanism to load params for gan-tree
if resume_flag is not False:
    try:
        if args.resume is True:
            iter_no = model.load_params(dir_name='all', param_group='all')
        else:
            iter_no = int(args.resume)
            iter_no = model.load_params(dir_name='all', param_group='all', iter_no=iter_no)

        logger.info('Loading network weights from all-%d' % iter_no)
    except Exception as ex:
        traceback.print_exc()
        logger.error(ex)
        logger.error('Some Problem Occured while resuming... starting from iteration 1')
        raise Exception('Not found...')
else:
    iter_no = 0

if 'all' in args.delete or 'weights' in args.delete:
    logger.warning('Deleting all weights in {}...'.format(paths.all_weights_dir))
    bash_utils.delete_recursive(paths.all_weights_dir)
    model_utils.setup_dirs()
    print('')

n_step_console_log = 500
n_step_tboard_log = 10
max_epochs = 10000
n_step_validation = 50
n_step_iter_save = 5000
n_step_visualize = 1000
can_visualize = lambda: iter_no % n_step_visualize == 0 or (iter_no < 2000 and iter_no % 200 == 0)

n_step_generator = 5
n_step_generator_decay = 1000

en_loss_history = []
de_loss_history = []
di_loss_history = []
d_acc_history = []
g_acc_history = []
gen_loss_history = []


def get_x_plots_data(model, x_input):
    _, x_real_true, x_real_false = model.discriminate(x_input)

    z_real_true = model.encode(x_real_true)
    z_real_false = model.encode(x_real_false)

    x_recon = model.reconstruct_x(x_input)
    _, x_recon_true, x_recon_false = model.discriminate(x_recon)

    z_recon_true = model.encode(x_recon_true)
    z_recon_false = model.encode(x_recon_false)

    return [
        (x_real_true, x_real_false),
        (z_real_true, z_real_false),
        (x_recon_true, x_recon_false),
        (z_recon_true, z_recon_false)
    ]


def get_z_plots_data(model, z_input):
    x_input = model.decode(z_input)
    x_plots = get_x_plots_data(model, x_input)
    return [z_input] + x_plots[:-1]


def train_step(model, iter_no):
    if H.train_autoencoder:
        model.step_train_autoencoder(train_inputs)

    if (iter_no % 40) < 10:
        if H.train_generator_adv:
            model.step_train_adv_generator(train_inputs)
    else:
        model.step_train_discriminator(train_inputs)


z_test = dl.get_z_dist(x_test.shape[0], dist_type=H.z_dist_type)
n_step_refresh_cluster_labels = 100

# Train the Root Node till saturation
node_train_iter = 0
gnode = gtree.root
X_splits = {0: x_train}


def refresh_splits(node):
    x_train_splits, _ = node.split_x(X_splits[node.node_id])
    X_splits.update(x_train_splits)
    return x_train_splits


def get_next_split(x_splits, ganset):
    # type: (dict, GANSet) -> int
    likelihoods = {gset[i].node_id: gset[i] for i in range(ganset.size)}
    min_likelihood_node_id = min(likelihoods.keys(), key=lambda l: likelihoods[l])
    return min_likelihood_node_id


while node_train_iter < max_epochs:
    node_train_iter += 1

    z_train = dl.get_z_dist(x_train.shape[0], dist_type=H.z_dist_type, bounds=H.z_bounds)

    train_inputs = np_utils.shuffled_copy(x_train), z_train

    train_step(gnode.model, node_train_iter)

    # TODO: validation, summary logging, visualization, and weight saving

for i_modes in range(1, H.n_modes + 1):
    # TODO: Find out the next node to split
    gset = gtree.get_gans(i_modes)
    next_split_node = None  # type: gan_tree.GNode

    parent_node = next_split_node
    new_nodes = gtree.split_node(next_split_node)
    cluster_ids = parent_node.child_node_ids
    x_train_splits = refresh_splits(parent_node)

    node_train_iter = 0

    while node_train_iter < max_epochs:
        node_train_iter += 1

        cluster_id = np.random.choice(cluster_ids, p=parent_node.cluster_probs)

        current_node = parent_node.get_child(cluster_id)

        x_train = np_utils.shuffled_copy(X_splits[cluster_id])
        z_train = current_node.sample_z_batch(x_train.shape[0])

        train_inputs = x_train, z_train

        train_step(current_node.model, node_train_iter)

        if node_train_iter % n_step_refresh_cluster_labels == 0:
            x_train_splits, _ = parent_node.split_x(X_splits[parent_node.node_id])
            X_splits.update(x_train_splits)

        # TODO: validation, summary logging, visualization, and weight saving

        # train_losses = model.compute_losses(train_inputs, model.network_loss_variables)
        # #
        # if iter_no % n_step_console_log == 0:
        #     print('Step %i: Encoder Loss: %f' % (iter_no, train_losses[0]))
        #     print('Step %i: Disc Acc: %f' % (iter_no, train_losses[1]))
        #     print('Step %i: Gen  Acc: %f' % (iter_no, train_losses[2]))
        #     print('Step %i: x_recon Loss: %f' % (iter_no, train_losses[3]))
        #     print('Step %i: z_recon Loss: %f' % (iter_no, train_losses[4]))
        #     print()
        #
        # if iter_no % n_step_tboard_log == 0:
        #     model.get_logger('train').add_summary(train_losses[-1], iter_no)
        #
        # if iter_no % n_step_validation == 0:
        #     test_losses = model.compute_losses(test_inputs, model.network_loss_variables)
        #     model.get_logger('test').add_summary(test_losses[-1], iter_no)
        #
        # if iter_no % n_step_iter_save == 0:
        #     model.save_params(iter_no=iter_no)
        #
        # if H.show_visual_while_training and can_visualize(iter_no):
        #     x_full = dl.get_full_space()
        #     x_plots_row1 = get_x_plots_data(model, x_test)
        #     z_plots_row2 = get_z_plots_data(model, z_test)
        #     x_plots_row3 = get_x_plots_data(model, x_full)
        #     plots_data = (x_plots_row1, z_plots_row2, x_plots_row3)
        #     figure = viz_utils.get_figure(plots_data)
        #     figure_name = 'plots-iter-%d.png' % iter_no
        #     figure_path = paths.get_result_path(figure_name)
        #     figure.savefig(figure_path)
        #     plt.close(figure)
        #     img = plt.imread(figure_path)
        #     model.log_image('test', img, iter_no)
        #
        # iter_time_end = time.time()
        #
        # if iter_no % n_step_console_log == 0:
        #     print('Single Iter Time: %.4f' % (iter_time_end - iter_time_start))
