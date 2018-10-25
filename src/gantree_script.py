
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

from trainers.gan_trainer import get_x_plots_data, get_z_plots_data

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
    0: {'x': x_1_complete[:128]},
    1: {'x': x_2_complete[:128]}
}

dl1 = child_loader(name='node1', data=x_1_complete)
dl2 = child_loader(name='node2', data=x_2_complete)

x_dl = [dl1, dl2]

# TODO: node1 and node2 from node0

E_0 = gan.encoder.copy()

d_1 = gan.decoder.copy()
d_2 = gan.decoder.copy()

di_1 = gan.disc.copy()
di_2 = gan.disc.copy()

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
