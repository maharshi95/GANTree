import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def twin_scatter(ax, data, colors=('green', 'red')):
    for i in range(len(colors)):
        ax.scatter(data[i][:, 0], data[i][:, 1], c=colors[i], s=3)


def plot_points(ax, data):
    if H.input_size == 1:
        twin_hist(ax, data, colors=('green', 'red'))
    elif H.input_size == 2:
        twin_scatter(ax, data, ('green', 'red'))


def get_figure(data, scatter_size=5):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, fig)
    # gs.update(wspace=0.025, hspace=0.05)

    rows = data

    for row_i in [0, 2]:
        x_real, z_real, x_recon, z_recon = rows[row_i]

        # lim_lower = np.minimum(np.min(x_real, axis=0), np.min(x_recon, axis=0))
        # lim_upper = np.maximum(np.max(x_real, axis=0), np.max(x_recon, axis=0))
        # length = max(lim_upper - lim_lower)
        # center = (lim_lower + lim_upper) / 2.0
        # lim_lower = center - length / 2.0
        # lim_upper = center + length / 2.0

        ax = [None] * 4
        ax[0] = plt.subplot(gs[row_i * 4 + 0])
        # ax[0].set_xlim(lim_lower[0], lim_upper[0])
        # ax[0].set_ylim(lim_lower[0], lim_upper[0])
        plot_points(ax[0], x_real)

        ax[1] = plt.subplot(gs[row_i * 4 + 1])

        plot_points(ax[1], z_real)

        ax[2] = plt.subplot(gs[row_i * 4 + 2])
        # ax[2].set_xlim(-2, 2)
        # ax[2].set_ylim(-2, 2)
        plot_points(ax[2], x_recon)

        ax[3] = plt.subplot(gs[row_i * 4 + 3])
        # ax[3].set_xlim(-1, 1)
        plot_points(ax[3], z_recon)

    for row_i in [1]:
        z_rand, x_real, z_real, x_recon = rows[row_i]
        ax = [None] * 4

        ax[0] = plt.subplot(gs[row_i * 4 + 0])
        if z_rand.shape[-1] == 1:
            ax[0].hist(z_rand, bins=50, ec='black')
        elif z_rand.shape[-1] == 2:
            ax[0].scatter(z_rand[:, 0], z_rand[:, 1], s=3)

        ax[1] = plt.subplot(gs[row_i * 4 + 1])
        # ax[1].set_xlim(-2, 2)
        # ax[1].set_ylim(-2, 2)
        plot_points(ax[1], x_real)

        ax[2] = plt.subplot(gs[row_i * 4 + 2])
        # ax[2].set_xlim(-1, 1)
        plot_points(ax[2], z_real)

        ax[3] = plt.subplot(gs[row_i * 4 + 3])
        # ax[3].set_xlim(-2, 2)
        # ax[3].set_ylim(-2, 2)
        plot_points(ax[3], x_recon)
    return fig
