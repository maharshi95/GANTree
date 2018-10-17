import os, argparse, logging, json

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

resume_flag = args.resume is not False

from exp_context import ExperimentContext

ExperimentContext.set_context(args.hyperparams, args.exp_name)
H = ExperimentContext.Hyperparams  # type: Hyperparams

logger = logging.getLogger(__name__)
LOG_FORMAT = "[{}: %(filename)s: %(lineno)3s] %(levelname)s: %(funcName)s(): %(message)s".format(ExperimentContext.exp_name)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

gpu_idx = str(args.gpu)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

import paths
from utils import bash_utils, model_utils

if 'all' in args.delete or 'logs' in args.delete or resume_flag is False:
    logger.warning('Deleting Logs...')
    bash_utils.delete_recursive(paths.logs_base_dir)
    print('')

if 'all' in args.delete or 'results' in args.delete:
    logger.warning('Deleting all results in {}...'.format(paths.results_base_dir))
    bash_utils.delete_recursive(paths.results_base_dir)
    print('')

model_utils.setup_dirs()

from dataloaders.factory import DataLoaderFactory
from base.hyperparams import Hyperparams

from models.toy.nets import ToyGAN
from trainers.gan_trainer import GanTrainer
from trainers.gan_trainer import TrainConfig

train_config = TrainConfig(
    n_step_tboard_log=100,
    n_step_console_log=100,
    n_step_validation=200,
    n_step_save_params=200,
    n_step_visualize=200
)

gan = ToyGAN('gan')
dl = DataLoaderFactory.get_dataloader(H.dataloader, H.input_size, H.z_size, H.batch_size, H.batch_size)
trainer = GanTrainer(data_loader=dl, model=gan, hyperparams=H, train_config=train_config)

hyperparams_string_content = json.dumps(H.__dict__, default=lambda x: repr(x), indent=4, sort_keys=True)
print(hyperparams_string_content)
with open(paths.exp_hyperparams_file, "w") as fp:
    fp.write(hyperparams_string_content)

trainer.train()

"""
snippets of gantree
"""


class child_loader(object):
    def __init__(self, name, data, batch_size=64):
        self.x = data,
        self.batch_size = batch_size
        self.name = name
        self.data = data
        self.n_batches = self.data.shape[0] // self.batch_size

        self.batch_index = 0

    def next_batch(self):
        start = self.batch_index * self.batch_size
        end = start + self.batch_size
        self.batch_index = (self.batch_size + 1) % self.n_batches

        if self.batch_index == 0:
            np.random.shuffle(self.data)

        data = self.data[start: end]
        return data


def sample(node1, node2):
    val = np.random.uniform()
    prob1 = node1.prob
    if val < prob1:
        return 0
    return 1


import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn import mixture
from collections import namedtuple
import numpy as np
from torch import distributions as dist
import torch as tr
from tensorboardX import SummaryWriter
import paths
from utils import viz_utils
import imageio as im


def get_x_plots_data(model, x_input):
    _, x_real_true, x_real_false = model.discriminate(x_input)

    z_real_true = model.encode(x_real_true)
    z_real_false = model.encode(x_real_false)

    x_recon = model.reconstruct_x(x_input)
    _, x_recon_true, x_recon_false = model.discriminate(x_recon)

    z_recon_true = model.encode(x_recon_true)
    z_recon_false = model.encode(x_recon_false)

    return [
        (x_real_true.numpy(), x_real_false.numpy()),
        (z_real_true.numpy(), z_real_false.numpy()),
        (x_recon_true.numpy(), x_recon_false.numpy()),
        (z_recon_true.numpy(), z_recon_false.numpy())
    ]


def get_z_plots_data(model, z_input):
    x_input = model.decode(z_input)
    x_plots = get_x_plots_data(model, x_input)
    return [z_input.numpy()] + x_plots[:-1]


child_params = namedtuple(
    'child_params',
    'mean '
    'cov '
    'con_prob '
    'prob'
)
writer = {
    0: SummaryWriter(paths.log_writer_path('child0')),
    1: SummaryWriter(paths.log_writer_path('child1')),
}

x_complete = dl.train_data()

# TODO: gmm and split code snippet

node0_gmm = mixture.GaussianMixture(n_components=H.n_child_nodes, covariance_type='full', max_iter=1000)
z_batch = gan.encode(x_complete)

_ = node0_gmm.fit(z_batch)
predictions = node0_gmm.predict(z_batch)
x_1 = []
x_2 = []
for ind, val in enumerate(predictions):
    if val == 0:
        x_1.append(x_complete[ind])
    else:
        x_2.append(x_complete[ind])



x_1_complete = tr.stack(x_1)
x_2_complete = tr.stack(x_2)

test_seed = {
    0: {'x': x_1_complete[:128]

        },
    1: {'x': x_2_complete[:128]

        }
}

dl1 = child_loader(name='node1', data=x_1_complete)
dl2 = child_loader(name='node2', data=x_2_complete)

x_dl = [dl1, dl2]

# TODO: node1 and node2 from node0

E_0 = gan.encoder.copy(gan.encoder)

d_1 = gan.decoder.copy(gan.decoder)
d_2 = gan.decoder.copy(gan.decoder)

di_1 = gan.disc.copy(gan.disc)
di_2 = gan.disc.copy(gan.disc)

node_1 = ToyGAN(name='node1', encoder=E_0, decoder=d_1, disc=di_1)
node_2 = ToyGAN(name='node2', encoder=E_0, decoder=d_2, disc=di_2)

child_models = [node_1, node_2]

child = []
for i in range(2):
    node = child_params(
        mean=(tr.from_numpy(node0_gmm.means_[i])).float(),
        cov=(tr.from_numpy(node0_gmm.covariances_[i])).float(),
        con_prob=node0_gmm.weights_[i],
        prob=1 * node0_gmm.weights_[i])

    child.append(node)

iter = 0
n_step_console_log = 2
board_log_step = 10
visualization_step = 20
params_save_step = 20
iterations = [0, 0]

while iter < 2 * H.n_iterations:
    iter += 1
    node_idx = sample(child[0], child[1])

    node = child[node_idx]
    model = child_models[node_idx]

    x = x_dl[node_idx].next_batch()
    x = x.float()
    # mvn = dist.MultivariateNormal(node.mean, node.cov)

    x1 = x_dl[0].next_batch()
    x2 = x_dl[1].next_batch()
    x1 = x1.float()
    x2 = x2.float()
    x2 = x2.float()

    child_iter = 0
    n_iter_gen = 0
    n_iter_disc = 0
    train_generator = True

    while child_iter < H.child_iter:
        child_iter += 1
        # z = mvn.sample_n(x.shape[0])
        z = np.random.multivariate_normal(node.mean, node.cov, x.shape[0])
        z = tr.from_numpy(z)

        # print 'z_shape', z.shape
        # print 'x1', x1.shape
        # print 'x2', x2.shape

        iterations[node_idx] += 1

        z = z.float()
        if H.train_autoencoder:
            model.step_train_autoencoder(x, z)
            model.step_train_x_clf(x1, x2, child[0].mean, child[1].mean, child[0].cov, child[1].cov)

        if train_generator:
            n_iter_gen += 1
            if H.train_generator_adv:
                model.step_train_generator(z)
        else:
            n_iter_disc += 1
            model.step_train_discriminator(x, z)

        metrics = model.compute_metrics(x, z)
        g_acc, d_acc = metrics['accuracy_gen'], metrics['accuracy_disc']

        if child_iter % n_step_console_log == 0:
            print "child-%d" % node_idx
            print('Train Step ', child_iter + 1)

            print('Step %i: Disc Acc: %f' % (child_iter + 1, metrics['accuracy_disc'].item()))
            print('Step %i: Gen  Acc: %f' % (child_iter + 1, metrics['accuracy_gen'].item()))
            print('Step %i: x_recon Loss: %f' % (child_iter + 1, metrics['loss_x_recon'].item()))
            print('Step %i: z_recon Loss: %f' % (child_iter + 1, metrics['loss_z_recon'].item()))
            print()

        if child_iter % board_log_step:
            for tag, value in metrics.items():
                writer[node_idx].add_scalar(tag, value.item(), iterations[node_idx])

        if child_iter % params_save_step:
            model.save_params(dir_name='iter', weight_label='iter', iter_no=sum(iterations))

        if child_iter % visualization_step:
            x_full = dl.get_full_space()

            x_plots_row1 = get_x_plots_data(model, test_seed[node_idx]['x'])
            z_plots_row2 = get_z_plots_data(model, model.encode(test_seed[node_idx]['x']))
            x_plots_row3 = get_x_plots_data(model, x_full)

            plots_data = (x_plots_row1, z_plots_row2, x_plots_row3)

            figure = viz_utils.get_figure(plots_data)
            figure_name = 'plots-iter-%d.png' % child_iter
            figure_path = paths.get_result_path(figure_name)
            figure.savefig(figure_path)
            plt.close(figure)

            img = np.array(im.imread(figure_path), dtype=np.uint8)
            img = img[:, :, :-1]
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            # image = tr.from_numpy(img)
            writer[node_idx].add_image('plot_iter', img, iterations[node_idx])

        if train_generator:
            if n_iter_gen == H.gen_iter_count or g_acc >= 70:
                train_generator = False

        if not train_generator:
            if n_iter_disc == H.disc_iter_count or d_acc >= 95:
                n_iter_disc = 0
                train_generator = True

    z1 = child_models[0].encode(x_1_complete)
    z2 = child_models[1].encode(x_2_complete)

    gmm1 = mixture.GaussianMixture(1)
    gmm1.fit(z1)

    gmm2 = mixture.GaussianMixture(1)
    gmm2.fit(z2)

    for ind, gmm in enumerate([gmm1, gmm2]):
        new_node = child_params(
            mean=(tr.from_numpy(gmm.means_)).float()[-1],
            cov=(tr.from_numpy(gmm.covariances_)).float()[-1],
            con_prob=node0_gmm.weights_[ind],
            prob=1 * node0_gmm.weights_[ind])

        child[ind] = new_node
        print ("modified child-node %d params " % (ind))

    print child
