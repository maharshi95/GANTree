import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from exp_context import ExperimentContext

H = ExperimentContext.Hyperparams


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
    n_labels = len(set(labels))
    colors = cm.rainbow(np.linspace(0, 1, n_labels))
    ax.scatter(data[:, 0], data[:, 1], c=colors[labels], s=scatter_size)


def get_axes(gs_item, axes3d=False):
    # type: (object, object) -> object
    if axes3d:
        ax = plt.subplot(gs_item, projection='3d')
    else:
        ax = plt.subplot(gs_item)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    return ax


def get_figure(data, scatter_size=3, margin=0.05):
    n_rows, n_cols = 4, 6
    fig = plt.figure(figsize=(n_cols * 2, n_rows * 2))
    gs = gridspec.GridSpec(n_rows, n_cols, fig)
    gs.update(bottom=margin, left=margin, top=1 - margin, right=1 - margin)

    rows = data

    for i in [0, 3]:
        x_real, z_real, zt_real, x_recon, z_recon, zt_recon = rows[i]
        x_3d = x_real[0].shape[-1] == 3
        z_3d = z_real[0].shape[-1] == 3
        _3d = [x_3d, z_3d, z_3d, x_3d, z_3d, z_3d]

        ax = [None] * n_cols
        for j, data in enumerate(rows[i]):
            ax[j] = get_axes(gs[i * n_cols + j], _3d[j])
            plot_points(ax[j], data, scatter_size)

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

    return fig
