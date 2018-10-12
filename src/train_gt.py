from __future__ import division, print_function, absolute_import

import json
import os, time, argparse, logging, traceback

# LOG_FORMAT = "[%(filename)s: %(lineno)3s] %(levelname)s: %(message)s"
# logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

LOG_FORMAT = "[{}: %(filename)s: %(lineno)3s] %(levelname)s: %(funcName)s(): %(message)s".format(ExperimentContext.exp_name)

if any(map(lambda obj: isinstance(obj, logging.StreamHandler), logger.handlers)):
    handler = filter(lambda obj: isinstance(obj, logging.StreamHandler), logger.handlers)[0]
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

import paths
from utils import viz_utils, np_utils

print(dir(np_utils))
from utils import bash_utils, model_utils
from dataloaders.factory import DataLoaderFactory

from models.bcgnode import BaseModel
from gan_tree.gan_tree_v2 import GANSet
from gan_tree import gan_tree_v2 as gan_tree

# GPU Selection
gpu_idx = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

# Check if delete Logs
resume_flag = args.resume is not False

if 'all' in args.delete or 'logs' in args.delete or resume_flag is False:
    logger.warning('Deleting Logs...')
    bash_utils.delete_recursive(paths.logs_base_dir)
    print('')

# Check if delete Results
if 'all' in args.delete or 'results' in args.delete:
    logger.warning('Deleting all results in {}...'.format(paths.results_base_dir))
    bash_utils.delete_recursive(paths.results_base_dir)
    print('')

model_utils.setup_dirs()

# Init Model, Hyperparams and DataLoader
Model = ExperimentContext.get_model_class()
H = ExperimentContext.get_hyperparams()
dl = DataLoaderFactory.get_dataloader(H.dataloader, H.input_size, H.z_size)

# Save Hyperparams into experiment directory
hyperparams_string_content = json.dumps(H.__dict__, default=lambda x: repr(x), indent=4, sort_keys=True)
print(hyperparams_string_content)
with open(paths.exp_hyperparams_file, "w") as fp:
    fp.write(hyperparams_string_content)

# Create, Build and Initiate GanTree
gtree = gan_tree.GanTree('GT', Model)
gtree.initiate()
print('GAN Tree service initiated...Current active nodes: {}'.format(gtree.n_active_nodes))

## TODO: Decide the mechanism to load params for gan-tree
if resume_flag is not False:
    logger.error('resume Flag is ON, this logic is not implemented yet. Execute the program without passing -r flag.')
    logger.info('Exiting program...')
    exit(-1)

    # try:
    #     if args.resume is True:
    #         iter_no = model.load_params(dir_name='all', param_group='all')
    #     else:
    #         iter_no = int(args.resume)
    #         iter_no = model.load_params(dir_name='all', param_group='all', iter_no=iter_no)
    #
    #     logger.info('Loading network weights from all-%d' % iter_no)
    # except Exception as ex:
    #     traceback.print_exc()
    #     logger.error(ex)
    #     logger.error('Some Problem Occured while resuming... starting from iteration 1')
    #     raise Exception('Not found...')
else:
    iter_no = 0

# Check if delete weights
if 'all' in args.delete or 'weights' in args.delete:
    logger.warning('Deleting all weights in {}...'.format(paths.all_weights_dir))
    bash_utils.delete_recursive(paths.all_weights_dir)
    model_utils.setup_dirs()
    print('')

# Define Intervals for each task
max_train_iterations = 20000
n_step_console_log = 200
n_step_tboard_log = 10
n_step_validation = 50
n_step_iter_save = 2000
n_step_visualize = 1000
can_visualize = lambda iter_no: iter_no % n_step_visualize == 0 or (iter_no < 2000 and iter_no % 200 == 0)

n_step_refresh_cluster_labels = 100

max_gen_accuracy = 60.0
max_disc_accuracy = 95.0
max_gen_steps = 10
max_disc_steps = 20
train_generator = True

gen_train_iter_count = 0
disc_train_iter_count = 0

# Init collectors for metrics
en_loss_history = []
de_loss_history = []
di_loss_history = []
d_acc_history = []
g_acc_history = []
gen_loss_history = []


####### PLOT FUNCTIONS ########

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


def get_random_batch(data, size=H.batch_size):
    perm = np.random.permutation(data.shape[0])[:size]
    return data[perm]


def get_random_z_batch():
    return dl.get_z_dist(H.batch_size, dist_type=H.z_dist_type, bounds=H.z_bounds)


def train_step(model, train_inputs, iter_no):
    global gen_train_iter_count, disc_train_iter_count, train_generator

    if H.train_autoencoder:
        model.step_train_autoencoder(train_inputs)
    network_name = 'Generator' if train_generator else 'Discriminator'
    # Select the network training
    if train_generator:
        # When gen_adv_training is kept off, it will switch to disc in next iteration.
        disc_acc, gen_acc = 0., 100.
        if H.train_generator_adv:
            disc_acc, gen_acc = model.step_train_adv_generator(train_inputs)
        gen_train_iter_count += 1
    else:
        disc_acc, gen_acc = model.step_train_discriminator(train_inputs)
        disc_train_iter_count += 1

    # Update next network to be trained
    if train_generator:
        if gen_train_iter_count == max_gen_steps or gen_acc < max_gen_accuracy:
            train_generator = False
    elif H.train_generator_adv:
        if disc_train_iter_count == max_disc_steps or disc_acc > max_disc_accuracy:
            train_generator = True

    if iter_no % n_step_console_log == 0:
        logger.info('Iter no: %d' % iter_no)
        logger.info('Training %s' % network_name)
        logger.info('Disc Acc: %.03f' % disc_acc)
        logger.info('Gen  Acc: %.03f' % gen_acc)
        print('')


# Fetch Data
x_train, x_test = dl.get_data()
x_data = {
    'train': x_train,
    'test': x_test
}

z_test = dl.get_z_dist(H.batch_size, dist_type=H.z_dist_type, bounds=H.z_bounds)
print('Train Test Data loaded...')

z_test_seed = get_random_batch(z_test, 10)
x_test_seed = get_random_batch(x_test, 10)

# Train the Root Node till saturation
global_iter_no = 0
node_iter_no = 0
gtree.x_batch = x_test
gnode = gtree.root
X_train_splits = {0: x_train}
X_test_splits = {0: x_test}


def refresh_splits(node):
    x_train_splits, _ = node.split_x(X_train_splits[node.id])
    x_test_splits, _ = node.split_x(X_test_splits[node.id])
    X_train_splits.update(x_train_splits)
    X_test_splits.update(x_test_splits)


def get_next_split_node(x_splits, gset):
    # type: (dict, gan_tree.GANSet) -> gan_tree.GNode
    likelihoods = {gset[i].node_id: gset[i].mean_likelihood(x_splits[gset[i].node_id]) for i in range(gset.size)}
    min_likelihood_node_id = min(likelihoods.keys(), key=lambda l: likelihoods[l])
    node_id = min_likelihood_node_id
    return gtree.nodes[node_id]


for i_modes in range(H.n_modes):
    node_iter_no = 0

    num_train_iterations = max_train_iterations if i_modes == 0 else 2 * max_train_iterations

    while node_iter_no < num_train_iterations:
        node_iter_no += 1

        iter_time_start = time.time()

        parent_node = None

        if i_modes == 0:  # First Iteration, No split required.
            selected_node = gtree.root
            selected_node_id = selected_node.id
        else:
            gset = gtree.create_ganset(i_modes)
            parent_node = get_next_split_node(X_train_splits, gset)
            child_nodes = gtree.split_node(parent_node)
            child_node_ids = parent_node.child_node_ids
            refresh_splits(parent_node)

            selected_node_id = np.random.choice(child_node_ids, p=parent_node.cluster_probs)
            selected_node = parent_node.get_child(selected_node_id)

        model = selected_node.model  # type: BaseModel

        x_train_batch = get_random_batch(X_train_splits[selected_node_id], H.batch_size)
        z_train_batch = selected_node.sample_z_batch(H.batch_size)

        train_inputs = x_train_batch, z_train_batch

        train_step(model, train_inputs, node_iter_no)

        if i_modes > 0 and node_iter_no % n_step_refresh_cluster_labels == 0:
            refresh_splits(parent_node)

        # TODO: validation, summary logging, visualization, and weight saving

        train_losses = model.compute_losses(train_inputs, model.network_loss_variables)

        # Console Log Step
        if node_iter_no % n_step_console_log == 0:
            print('Step %i: Encoder Loss: %f' % (node_iter_no, train_losses[0]))
            print('Step %i: Disc Acc: %f' % (node_iter_no, train_losses[1]))
            print('Step %i: Gen  Acc: %f' % (node_iter_no, train_losses[2]))
            print('Step %i: x_recon Loss: %f' % (node_iter_no, train_losses[3]))
            print('Step %i: z_recon Loss: %f' % (node_iter_no, train_losses[4]))
            print()

        # Tensorboard Log Step
        if node_iter_no % n_step_tboard_log == 0:
            model.get_logger('train').add_summary(train_losses[-1], node_iter_no)

        # Validation Step
        if node_iter_no % n_step_validation == 0:
            x_test_batch = get_random_batch(X_test_splits[selected_node_id], H.batch_size)
            z_test_batch = selected_node.sample_z_batch(H.batch_size)
            test_inputs = x_test_batch, z_test_batch

            test_losses = model.compute_losses(test_inputs, model.network_loss_variables)
            model.get_logger('test').add_summary(test_losses[-1], node_iter_no)

        # Parameter Saving Step
        if node_iter_no % n_step_iter_save == 0:
            model.save_params(iter_no=node_iter_no)

        # Visualization Step
        if H.show_visual_while_training and can_visualize(node_iter_no):
            x_full = dl.get_full_space()
            x_plots_row1 = get_x_plots_data(model, x_test)
            z_plots_row2 = get_z_plots_data(model, z_test)
            x_plots_row3 = get_x_plots_data(model, x_full)
            plots_data = (x_plots_row1, z_plots_row2, x_plots_row3)
            # print(plots_data)
            figure = viz_utils.get_figure(plots_data)
            figure_name = 'plots-iter-%d.png' % node_iter_no
            bash_utils.create_dir(paths.get_result_path('', model_name=model.name))
            figure_path = paths.get_result_path(figure_name, model.name)
            figure.savefig(figure_path)
            plt.close(figure)
            img = plt.imread(figure_path)
            # model.log_image('test', img, iter_no)

        iter_time_end = time.time()

        if node_iter_no % n_step_console_log == 0:
            print('Single Iter Time: %.4f' % (iter_time_end - iter_time_start))
