import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_figure(data, scatter_size=5):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, fig)
    # gs.update(wspace=0.025, hspace=0.05)

    rows = data

    for row_i in [0, 2]:
        x_real, z_real, x_recon, z_recon = rows[row_i]
        ax = [None] * 4
        ax[0] = plt.subplot(gs[row_i * 4 + 0])
        ax[0].set_xlim(-2, 2)
        ax[0].set_ylim(-2, 2)
        ax[0].scatter(x_real[0][:, 0], x_real[0][:, 1], c='green', s=scatter_size)
        ax[0].scatter(x_real[1][:, 0], x_real[1][:, 1], c='red', s=scatter_size)
        # ax0.set_xticklabels([])
        # ax0.set_yticklabels([])
        # ax0.set_aspect('equal')

        ax[1] = plt.subplot(gs[row_i * 4 + 1])
        ax[1].set_xlim(-1, 1)
        ax[1].hist(z_real[0], bins=50, ec='black', color='green')
        ax[1].hist(z_real[1], bins=50, ec='black', color='red')

        ax[2] = plt.subplot(gs[row_i * 4 + 2])
        ax[2].set_xlim(-2, 2)
        ax[2].set_ylim(-2, 2)
        ax[2].scatter(x_recon[0][:, 0], x_recon[0][:, 1], c='green', s=scatter_size)
        ax[2].scatter(x_recon[1][:, 0], x_recon[1][:, 1], c='red', s=scatter_size)

        ax[3] = plt.subplot(gs[row_i * 4 + 3])
        ax[3].set_xlim(-1, 1)
        ax[3].hist(z_recon[0], bins=50, ec='black', color='green')
        ax[3].hist(z_recon[1], bins=50, ec='black', color='red')

    for row_i in [1]:
        z_rand, x_real, z_real, x_recon = rows[row_i]
        ax = [None] * 4

        ax[0] = plt.subplot(gs[row_i * 4 + 0])
        ax[0].hist(z_rand, bins=50, ec='black')

        ax[1] = plt.subplot(gs[row_i * 4 + 1])
        ax[1].set_xlim(-2, 2)
        ax[1].set_ylim(-2, 2)
        ax[1].scatter(x_real[0][:, 0], x_real[0][:, 1], c='green', s=scatter_size)
        ax[1].scatter(x_real[1][:, 0], x_real[1][:, 1], c='red', s=scatter_size)
        # ax0.set_xticklabels([])
        # ax0.set_yticklabels([])
        # ax0.set_aspect('equal')

        ax[2] = plt.subplot(gs[row_i * 4 + 2])
        ax[2].set_xlim(-1, 1)
        ax[2].hist(z_real[0], bins=50, ec='black', color='green')
        ax[2].hist(z_real[1], bins=50, ec='black', color='red')

        ax[3] = plt.subplot(gs[row_i * 4 + 3])
        ax[3].set_xlim(-2, 2)
        ax[3].set_ylim(-2, 2)
        ax[3].scatter(x_recon[0][:, 0], x_recon[0][:, 1], c='green', s=scatter_size)
        ax[3].scatter(x_recon[1][:, 0], x_recon[1][:, 1], c='red', s=scatter_size)

    return fig
