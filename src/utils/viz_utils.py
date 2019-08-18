import numpy as np

from matplotlib import pyplot as plt

from scipy.spatial import distance

from exp_context import ExperimentContext
from utils.tr_utils import as_np
import torch as tr

from paths import Paths

def visualize_embeddings(node, split, threshold, iter_no, phase = None):
    with tr.no_grad():
        if split == 'train':
            data = dl_set[node.id].data[split]
        elif split == 'test':
            data = x_seed

        Z = node.post_gmm_encode(data)

        labels = node.gmm_predict_test(Z, threshold).tolist()

        pca_z = PCA(n_components = 2)

        z_transformed = pca_z.fit_transform(Z)

        color = ['r', 'b', 'g']
        colors = [color[int(x)] for x in labels]

        b = 20
        fig = plt.figure(figsize=(6.5, 6.5))

        ax = fig.add_subplot(111)
        ax.set_xlim(-b, b)
        ax.set_ylim(-b, b)

        ax.scatter(z_transformed[:, 0], z_transformed[:, 1], s = 0.5, c = colors)


        node.trainer.writer[split].add_figure(node.name + '_' + phase +'_plots', fig, iter_no)

        path = Paths.get_result_path(node.name + '_' + split + '_embedding_plots/'+ phase + '_plot_%03d' % (iter_no))
        fig.savefig(path)
        plt.close(fig)


def visualize_images(node, split, iter_no, phase):
    with tr.no_grad():
        if split == 'train':
            data = dl_set[node.id].data['train'][:1000]
        elif split == 'test':
            data = x_seed

        preds = node.gmm_predict(node.post_gmm_encode(data))
        x_data_child0 = data[np.where(preds == 0)].cuda()
        x_data_child1 = data[np.where(preds == 1)].cuda()

        if x_data_child0.shape[0] > 64:
            x_data_child0 = x_data_child0[:64]

        if x_data_child1.shape[0] > 64:
            x_data_child1 = x_data_child1[:64]

        if x_data_child0.shape[0] == 0:
            x_data_child0 = node.trainer.seed_data[split]['x']

        if x_data_child1.shape[0] == 0:
            x_data_child1 = node.trainer.seed_data[split]['x']

        z_data_child0 = node.get_child(0).gan.sample((x_data_child0.shape[0],))
        z_data_child1 = node.get_child(1).gan.sample((x_data_child1.shape[0],))

        x_recon_child0 = node.get_child(0).gan.reconstruct_x(x_data_child0)
        x_recon_child1 = node.get_child(1).gan.reconstruct_x(x_data_child1)
        x_gen_child0 = node.get_child(0).gan.decode(z_data_child0)
        x_gen_child1 = node.get_child(1).gan.decode(z_data_child1)

        recon_img_child0 = save_image(x_recon_child0)
        gen_img_child0 = save_image(x_gen_child0)
        recon_img_child1 = save_image(x_recon_child1)
        gen_img_child1 = save_image(x_gen_child1)
        real_img_child0 = save_image(x_data_child0)
        real_img_child1 = save_image(x_data_child1)

        node.trainer.writer[split].add_image(node.name + '_' + phase + '_child0_recon', recon_img_child0, iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child0_gen', gen_img_child0, iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child0_real', real_img_child0, iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child1_recon', recon_img_child1, iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child1_gen', gen_img_child1, iter_no)
        node.trainer.writer[split].add_image(node.name + '_' + phase +  '_child1_real', real_img_child1, iter_no)


def z_histogram_plot(node, split, iter_no, phase):
    with tr.no_grad():
        if split == 'train':
            data = dl_set[node.id].data[split]
        elif split == 'test':
            data = x_seed

        Z = node.post_gmm_encode(data)

        for i in range(Z.shape[1]):
            plot_data = Z[:, i]
            plt.hist(plot_data)

            fig_histogram = plt.gcf()
            node.trainer.writer[split].add_histogram(node.name + '_' + phase + '_embedding_' + str(i), plot_data, iter_no)
            path_embedding_hist = Paths.get_result_path(node.name + '_' + split +  '_embedding_histogram/' + phase + 'embedding_%03d_%01d' % (iter_no, i))
            fig_histogram.savefig(path_embedding_hist)
            plt.close(fig_histogram)


def get_labels_distribution(node, split):
    iter_no = 0
    with tr.no_grad():

        if split == 'train':
            data = dl_set[node.id].data[split]
            labels = dl_set[node.id].labels[split]
        elif split == 'test':
            data = x_seed
            labels = l_seed

        Z = node.post_gmm_encode(data)
        
        pred = node.gmm_predict(Z)

        labels_ch0 = labels[np.where(pred == 0)]
        labels_ch1 = labels[np.where(pred == 1)]

        np.savez(node.name + '_' + split + '_child_labels', labels_ch0 = labels_ch0, labels_ch1 = labels_ch1)

        count_ch0 = [0 for i in range(10)]
        count_ch1 = [0 for i in range(10)]
        prob_ch0 = [0 for i in range(10)]
        prob_ch1 = [0 for i in range(10)] 

        for i in labels_ch0:
            count_ch0[i] += 1

        for i in labels_ch1:
            count_ch1[i] += 1

        for i in range(10):
            if (count_ch0[i] + count_ch1[i]) != 0:
                prob_ch0[i] = count_ch0[i] * 1.0 / (count_ch0[i] + count_ch1[i])
                prob_ch1[i] = count_ch1[i] * 1.0 / (count_ch0[i] + count_ch1[i])
            else:
                prob_ch0[i] = 0
                prob_ch1[i] = 0

        barWidth = 0.3
        r1 = np.arange(len(count_ch0))
        r2 = [x+barWidth for x in r1]

        plt.bar(r1, prob_ch0, width = barWidth, color = 'red', edgecolor = 'black', capsize=7)
        plt.bar(r2, prob_ch1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7)
        plt.xticks([r + barWidth for r in range(len(prob_ch0))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.ylabel('percentage')

        fig_labels_prob = plt.gcf()
        node.trainer.writer[split].add_figure(node.name + '_labels_prob', fig_labels_prob, iter_no)
        path_labels_prob = Paths.get_result_path(node.name + '_' + split + '_labels_distribution/probability_%03d' % (iter_no))
        fig_labels_prob.savefig(path_labels_prob)
        plt.close(fig_labels_prob)


        plt.bar(r1, count_ch0, width = barWidth, color = 'red', edgecolor = 'black', capsize=7)
        plt.bar(r2, count_ch1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7)
        plt.xticks([r + barWidth for r in range(len(count_ch0))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.ylabel('count')

        fig_labels_count = plt.gcf()
        node.trainer.writer[split].add_figure(node.name + '_labels_distribution', fig_labels_count, iter_no)
        path_labels_count = Paths.get_result_path(node.name + '_' + split +  '_labels_distribution/count_%03d' % (iter_no))
        fig_labels_count.savefig(path_labels_count)
        plt.close(fig_labels_count)




def plot_cluster_graphs(node, split, threshold, iter_no, phase):

    with tr.no_grad():
        if split == 'train':
            data = dl_set[node.id].data[split]
            labels = dl_set[node.id].labels[split]
        elif split == 'test':
            data = x_seed
            labels = l_seed

        Z = node.post_gmm_encode(data)

        if split == 'train':
            p = node.kmeans.pred
        elif split == 'test':
            p = node.gmm_predict_test(Z, threshold)

        """ plot the count of unassigned vs assigned labels
            purple -- unassigned
            green -- assigned """

        unassigned_labels = [0 for i in range(3)]
        assigned_labels = [0 for i in range(3)]

        for i in range(len(p)):
            if p[i] == 2:
                unassigned_labels[labels[i]] += 1
            else:
                assigned_labels[labels[i]] += 1

        barWidth = 0.3
        r1 = np.arange(len(unassigned_labels))
        r2 = [x+barWidth for x in r1]

        plt.bar(r1, unassigned_labels, width = barWidth, color = 'purple', edgecolor = 'black', capsize=7)
        plt.bar(r2, assigned_labels, width = barWidth, color = 'green', edgecolor = 'black', capsize=7)
        plt.xticks([r + barWidth for r in range(len(unassigned_labels))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.ylabel('count')

        fig_assigned = plt.gcf()
        node.trainer.writer[split].add_figure(node.name + '_' + phase + '_assigned_labels_count', fig_assigned, iter_no)
        path_assign = Paths.get_result_path(node.name + '_' + split +  '_assigned/' + phase + 'assigned_%03d' % (iter_no))
        fig_assigned.savefig(path_assign)
        plt.close(fig_assigned)



        """ plot the percentage of assigned labels in cluster 0 and cluster 1
            red -- cluster 0
            blue -- cluster 1 """

        l_seed_ch0 = labels[np.where(p == 0)]
        l_seed_ch1 = labels[np.where(p == 1)]

        count_ch0 = [0 for i in range(3)]
        count_ch1 = [0 for i in range(3)]
        prob_ch0 = [0 for i in range(3)]
        prob_ch1 = [0 for i in range(3)]

        for i in l_seed_ch0:
            count_ch0[i] += 1

        for i in l_seed_ch1:
            count_ch1[i] += 1

        for i in range(3):
            if (count_ch0[i] + count_ch1[i]) != 0:
                prob_ch0[i] = count_ch0[i] * 1.0 / (count_ch0[i] + count_ch1[i])
                prob_ch1[i] = count_ch1[i] * 1.0 / (count_ch0[i] + count_ch1[i])
            else:
                prob_ch0[i] = 0
                prob_ch1[i] = 0

        plt.bar(r1, prob_ch0, width = barWidth, color = 'red', edgecolor = 'black', capsize=7)
        plt.bar(r2, prob_ch1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7)
        plt.xticks([r + barWidth for r in range(len(prob_ch0))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.ylabel('percentage')

        fig_confidence = plt.gcf()
        node.trainer.writer[split].add_figure(node.name +  '_' + phase + '_confidence', fig_confidence, iter_no)
        path_confidence = Paths.get_result_path(node.name + '_' + split + '_confidence/' + phase + 'confidence_%03d' % (iter_no))
        fig_confidence.savefig(path_confidence)
        plt.close(fig_confidence)


        """ get count of points that exceed the threshold of phase 1 part 2 """

        aboveThresholdLabels_ch0 = [0 for i in range(3)]
        aboveThresholdLabels_ch1 = [0 for i in range(3)]

        # percentAbove_ch0 = [0 for i in range(3)]
        # percentAbove_ch1 = [0 for i in range(3)]

        for i in range(len(p)):
            if p[i] == 0:
                if (distance.mahalanobis(Z[i], node.kmeans.means[0], node.kmeans.covs[0])) > threshold:
                    aboveThresholdLabels_ch0[labels[i]] += 1
            elif p[i] == 1:
                if (distance.mahalanobis(Z[i], node.kmeans.means[1], node.kmeans.covs[1])) > threshold:
                    aboveThresholdLabels_ch1[labels[i]] += 1

        # for i in range(3):
        #     if (count_ch0[i]) != 0:
        #         percentAbove_ch0[i] = aboveThresholdLabels_ch0[i] * 1.0 / count_ch0[i]
        #     else:
        #         percentAbove_ch0[i] = 0

        #     if (count_ch1[i] != 0):
        #         percentAbove_ch1[i] = aboveThresholdLabels_ch1[i] * 1.0 / count_ch1[i]
        #     else:
        #         percentAbove_ch1[i] = 0

        plt.bar(r1, aboveThresholdLabels_ch0, width = barWidth, color = 'red', edgecolor = 'black', capsize=7)
        plt.bar(r2, aboveThresholdLabels_ch1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7)
        plt.xticks([r + barWidth for r in range(len(aboveThresholdLabels_ch0))], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.ylabel('count')

        fig_above_threshold = plt.gcf()
        node.trainer.writer[split].add_figure(node.name + '_' + phase + '_above_threshold', fig_above_threshold, iter_no)
        path_above_threshold = Paths.get_result_path(node.name + '_' + split + '_above_threshold/' + phase + '%03d' % (iter_no))
        fig_above_threshold.savefig(path_above_threshold)
        plt.close(fig_above_threshold)



def plot_mean_axis_distribution(node, split, iter_no, phase):

    mean0 = node.kmeans.means[0]
    mean1 = node.kmeans.means[1]

    direction = (mean1 - mean0) / np.linalg.norm(mean1 - mean0)

    if split == 'train':
        data = dl_set[node.id].data['train']
    elif split == 'test':
        data = x_seed

    Z = node.post_gmm_encode(data)

    projection = np.zeros(Z.shape)

    for j in range(Z.shape[0]):
        projection[j] = mean0 + direction * np.dot(Z[j] - mean0, direction)

    for i in range(projection.shape[1]):
        plot_data_tensorboard = projection[:, i] 
        plot_data = [projection[:, i], mean0[i], mean1[i]]
        plt.hist(plot_data, color = ['g', 'r', 'b'])
        # plt.hist(plot_data_tensorboard, bins = 'auto', color = ['g'])

        fig_mean_axis_histogram = plt.gcf()
        node.trainer.writer[split].add_histogram(node.name + '_' + phase + '_mean_axis_' + str(i), plot_data_tensorboard, iter_no)
        # node.trainer.writer[split].add_image(node.name + '_mean_axis_' + str(i), fig_mean_axis_histogram, iter_no)
        path_mean_axis_hist = Paths.get_result_path(node.name + '_' + split +  '_mean_axis_histogram/' + phase + '%03d_%01d' % (iter_no, i))
        fig_mean_axis_histogram.savefig(path_mean_axis_hist)
        plt.close(fig_mean_axis_histogram)
