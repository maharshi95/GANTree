import collections
from sys import pydebug

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture

from exp_context import ExperimentContext
from utils import np_utils
from utils.tr_utils import as_np

H = ExperimentContext.Hyperparams

bounds = 10


def scatter_2d(ax, data, s=0.5, c=None, marker=None, linewidths=None, *args, **kwargs):
    return ax.scatter(data[:, 0], data[:, 1], s=s, c=c, marker=None, linewidths=linewidths, *args, **kwargs)


def plot_ellipse(ax, means, cov, scales=3.0, color='red'):
    e = get_ellipse(means, cov, scales, color=color)
    return ax.add_artist(e)


def get_random_colors(k):
    colors = cm.rainbow(np.linspace(0, 1, k))
    return colors


def get_ellipse(means, cov, scales=3.0, alpha=0.25, color='yellow'):
    if not isinstance(scales, collections.Iterable):
        center, theta, width, height = np_utils.ellipse_params(means, cov, scale=scales)
        return Ellipse(center, width, height, np.rad2deg(theta), alpha=alpha, color=color)

    ellipses = []
    for scale in scales:
        center, theta, width, height = np_utils.ellipse_params(means, cov, scale=scale)
        e = Ellipse(center, width, height, np.rad2deg(theta), alpha=alpha, color=color)
        ellipses.append(e)
    return ellipses


def plot_ellipses(ax, gmm, scales):
    n = gmm.n_components
    colors = get_random_colors(n)
    for i in range(n):
        ellipses = get_ellipse(gmm.means_[i], gmm.covariances_[i], scales, color=colors[i])
        if isinstance(ellipses, list):
            for e in ellipses:
                ax.add_artist(e)
        else:
            ax.add_artist(ellipses)


def hist(ax, X, binwidth=0.2, color=None):
    if X.shape[0] == 0:
        return
    bins = np.arange(np.min(X), np.max(X) + binwidth, binwidth)
    return ax.hist(X, bins, color=color, ec='black', normed=True)


def twin_hist(ax, X, binwidth=0.2, colors=('green', 'red')):
    hist(ax, X[0], binwidth=binwidth, color=colors[0])
    hist(ax, X[1], binwidth=binwidth, color=colors[1])


def twin_scatter(ax, data, colors=('green', 'red'), scatter_size=3):
    for i in range(len(colors)):
        if data[i].shape[-1] == 2:
            ax.scatter(data[i][:, 0], data[i][:, 1], c=colors[i], s=scatter_size)
        elif data[i].shape[-1] == 3:
            ax.scatter(data[i][:, 0], data[i][:, 1], data[i][:, 2], c=colors[i], s=scatter_size)


def plot_points(ax, data, scatter_size=3):
    if H.input_size == 1:
        twin_hist(ax, data, colors=('green', 'red'))
    elif H.input_size <= 3:
        twin_scatter(ax, data, ('green', 'red'), scatter_size)


def plot_clusters(ax, data, labels, scatter_size=3):
    labels = labels.astype(int)
    indices = map(lambda l: list(set(labels)).index(l), labels)
    n_labels = len(set(labels))
    colors = cm.rainbow(np.linspace(0, 1, n_labels))
    ax.scatter(data[:, 0], data[:, 1], c=colors[indices], s=scatter_size)


def get_axes(gs_item, axes3d=False):
    # type: (object, object) -> object
    if axes3d:
        ax = plt.subplot(gs_item, projection='3d')
    else:
        ax = plt.subplot(gs_item)
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    return ax


def get_figure(data, scatter_size=3, margin=0.05):
    n_rows, n_cols = 4, 6
    fig = plt.figure(figsize=(n_cols * 2, n_rows * 2))
    gs = gridspec.GridSpec(n_rows, n_cols, fig)
    gs.update(bottom=margin, left=margin, top=1 - margin, right=1 - margin)

    rows = data

    z_data, zt_data = rows[-1]
    gmm = GaussianMixture(2)
    gmmt = GaussianMixture(2)

    gmm.fit(z_data)
    gmmt.fit(zt_data)

    # X-Z-X-Z iteration with real and full space
    for i in [0, 3]:
        x_real, z_real, zt_real, x_recon, z_recon, zt_recon = rows[i]
        x_3d = x_real[0].shape[-1] == 3
        z_3d = z_real[0].shape[-1] == 3
        _3d = [x_3d, z_3d, z_3d, x_3d, z_3d, z_3d]

        ax = [None] * n_cols
        for j, data in enumerate(rows[i]):
            ax[j] = get_axes(gs[i * n_cols + j], _3d[j])
            plot_points(ax[j], data, scatter_size)

    # Z-X-Z-X iteration with real and full space
    for i in [1]:
        z_rand, x_real, z_real, zt_real, x_recon, z_recon = rows[i]
        x_3d = x_real[0].shape[-1] == 3
        z_3d = z_real[0].shape[-1] == 3
        _3d = [z_3d, x_3d, z_3d, z_3d, x_3d, z_3d]

        ax = [None] * len(rows[i])

        if z_rand.shape[-1] == 1:
            ax[0] = get_axes(gs[i * n_cols + 0])
            ax[0].hist(z_rand, bins=50, ec='black')
        elif z_rand.shape[-1] == 2:
            ax[0] = get_axes(gs[i * n_cols + 0])
            ax[0].scatter(z_rand[:, 0], z_rand[:, 1], s=scatter_size)
        elif z_rand.shape[-1] == 3:
            ax[0] = get_axes(gs[i * n_cols + 0], axes3d=True)
            ax[0].scatter(z_rand[:, 0], z_rand[:, 1], z_rand[:, 2], s=scatter_size)

        for j in range(1, n_cols):
            ax[j] = get_axes(gs[i * n_cols + j], _3d[j])
            plot_points(ax[j], rows[i][j], scatter_size)

    # X-Z-X-Z iteration with real x with cluster colours
    for i in [2]:
        row = rows[i]
        # x, z, zt, x_, z_, zt_, l = rows[i]
        l = row[-1]

        x_3d = x_real[0].shape[-1] == 3
        z_3d = z_real[0].shape[-1] == 3
        _3d = [x_3d, z_3d, z_3d, x_3d, z_3d, z_3d]

        ax = [None] * n_cols
        for j in range(n_cols):
            ax[j] = get_axes(gs[i * n_cols + j], _3d[j])

            plot_clusters(ax[j], row[j], l, scatter_size)

        plot_ellipses(ax[4], gmm, [1, 2, 3])
        plot_ellipses(ax[5], gmmt, [1, 2, 3])

    return fig


def get_x_clf_figure(plot_data, n_modes=9):
    [[
        [x_batch, labels],
        [root_means, root_cov, _, _],
        [ch0_means, ch0_cov, _, _],
        [ch1_means, ch1_cov, _, _]
    ], [
        z_batch_pre,
        z_batch_post,
        x_recon_pre,
        x_recon_post
    ], [
        z_rand0,
        x_fake0,
        z_rand1,
        x_fake1,

    ]] = plot_data
    b = 9
    colors = get_random_colors(n_modes)
    colors = colors[labels]

    fig = plt.figure(figsize=(12, 12))

    ax = fig.add_subplot(331)
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    scatter_2d(ax, x_batch, c=colors)

    ax = fig.add_subplot(332)
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    scatter_2d(ax, x_recon_post, c=colors)

    ax = fig.add_subplot(333)
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    scatter_2d(ax, z_batch_post, c=colors)
    plot_ellipse(ax, ch0_means, ch0_cov, color='red')
    plot_ellipse(ax, ch1_means, ch1_cov, color='blue')

    ax = fig.add_subplot(334)
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    scatter_2d(ax, x_batch)

    ax = fig.add_subplot(335)
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    scatter_2d(ax, x_recon_pre, c=colors)

    ax = fig.add_subplot(336)
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    scatter_2d(ax, z_batch_pre, c=colors)
    plot_ellipse(ax, root_means, root_cov, color='red')

    ax = fig.add_subplot(337)
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    scatter_2d(ax, z_rand0, c='red')
    scatter_2d(ax, z_rand1, c='blue')

    ax = fig.add_subplot(338)
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    scatter_2d(ax, x_fake0, c='red')
    scatter_2d(ax, x_fake1, c='blue')

    return fig
