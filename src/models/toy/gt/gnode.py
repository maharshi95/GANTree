from __future__ import absolute_import
import pickle
import time
from collections import Counter

import numpy as np
import torch as tr
from scipy import stats
from scipy.stats._multivariate import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.spatial import distance
from termcolor import colored
from torch import nn, optim

from models import losses
from models.toy.gan import ToyGAN
from trainers.gan_image_trainer import GanImgTrainer
from utils import np_utils, tr_utils, viz_utils
from utils.tr_utils import as_np
from .named_tuples import DistParams
from trainers.gan_trainer import GanTrainer
import logging

from configs import Config
from utils.decorators import make_tensor, tensorify

##########  Set Logging  ###########
logger = logging.getLogger(__name__)


# LOG_FORMAT = "[{}: %(filename)s: %(lineno)3s] %(levelname)s: %(funcName)s(): %(message)s".format(ExperimentContext.exp_name)
# logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class GMM(object):
    def __init__(self, means, cov, weights):
        self.means = means
        self.cov = cov
        self.weights = weights

class KMeansCltr(object):
    def __init__(self, means, covs, weights, pred, cluster_centers, pca):
        self.means = means
        self.covs = covs
        self.weights = weights
        self.pred = pred
        self.cluster_centers = cluster_centers
        self.pca = pca


class LeafNodeException(Exception):
    pass


def disallow_leafs(f):
    def inner(self, *args, **kwargs):
        if self.is_leaf:
            raise LeafNodeException('%s called on leaf node.' % f.__name__)
        return f(*args, **kwargs)

    return inner


def allow_only_leafs(f):
    def inner(self, *args, **kwargs):
        if not self.is_leaf:
            raise LeafNodeException('%s called on a non leaf node.' % f.__name__)
        return f(*args, **kwargs)

    return inner


class GNode(nn.Module):
    trainer = None  # type: GanImgTrainer
    opt_xc = None  # type: optim.Adam
    opt_xr = None  # type: optim.Adam
    gmm = None  # type: GaussianMixture
    kmeans = None # type: KMeansClusters
    child_nodes = None  # type: dict[int, GNode]

    @staticmethod
    def create_clone(node):
        # type: (GNode) -> GNode
        new_node = GNode(node_id=node.id, model=node.gan)
        new_node.gmm = node.gmm
        new_node.dist_params = DistParams(*node.dist_params)
        return new_node

    def __init__(self, node_id=-1, model=None, parent=None, dist_params=None):
        # type: (int, ToyGAN, GNode, DistParams) -> GNode
        super(GNode, self).__init__()
        self.id = node_id
        self.gan = model
        self.child_ids = []
        self.child_nodes = {}
        self.__assign_parent(parent)

        self.gmm = None
        self.trainer = None
        self.opt_xr = None
        self.opt_xc = None
        self.opt_xun = None
        self.opt_xrecon = None

        self.prior_means, self.prior_cov = model.z_op_params
        self.prior_prob = 1.
        self.prob = 1. if parent is None else parent.prob * self.prior_prob

    def __repr__(self):
        return '<GNode[name={} id={} parent_id={}]>'.format(self.name, self.id, self.parent_id)

    @property
    def name(self):
        return self.gan.name

    @property
    def model_class(self):
        return self.gan.__class__

    @property
    def is_root(self):
        return self.parent is None

    @property
    def is_leaf(self):
        return len(self.child_ids) == 0

    @property
    def parent_id(self):
        return -1 if self.parent is None else self.parent.id

    @property
    def n_child(self):
        return len(self.child_ids)

    @property
    def left(self):
        return self.get_child(0)

    @property
    def right(self):
        return self.get_child(1)

    @property
    def all_child(self):
        return [self.child_nodes[index] for index in self.child_nodes]

    @property
    def ellipse(self, color='red', scales=3.0):
        return viz_utils.get_ellipse(self.prior_means, self.prior_cov, scales=scales, color=color)

    @property
    def parent_name(self):
        return 'nullNode' if self.parent is None else self.parent.name

    @property
    def cluster_probs(self):
        return self.gmm.weights_

    @property
    def dist_params(self):
        return DistParams(self.prior_means, self.prior_cov, self.prior_prob, self.prob)

    @dist_params.setter
    def dist_params(self, params):
        # type: (tuple) -> None
        # tuple of 4 params
        self.prior_means, self.prior_cov, self.prior_prob, _ = params
        self.prob = self.prior_prob if self.parent is None else self.parent.prob * self.prior_prob

    @property
    def tensor_params(self):
        m, s, w, _ = self.dist_params
        return map(lambda v: tr.tensor(v, dtype=tr.float32), [m, s, w])

    def update_dist_params(self, means=None, cov=None, prior_prob=None):
        if means is not None:
            self.prior_means = means

        if cov is not None:
            self.prior_cov = cov

        if prior_prob is not None:
            self.prior_prob = prior_prob
            self.prob = self.prior_prob if self.parent is None else self.parent.prob * self.prior_prob

        self.gan.z_op_params = self.prior_means, self.prior_cov

    def get_child(self, index):
        # type: (int) -> GNode
        """
        Returns the child of the node at rank `index`
        """
        return self.child_nodes[self.child_ids[index]]

    @property
    def pre_gmm_encoder(self):
        return self.gan.encoder

    @property
    def pre_gmm_decoder(self):
        return self.gan.decoder

    @property
    def post_gmm_encoder(self):
        return self.get_child(0).gan.encoder

    @property
    def post_gmm_decoders(self):
        return [self.get_child(i).pre_gmm_decoder for i in range(self.n_child)]

    def pre_gmm_encode(self, X, transform=False, batch=128):
        Z = []
        n_batches = (X.shape[0] + batch - 1) / batch
        for i in range(n_batches):
            Z.append(self.gan.encode(X[i * batch:(i + 1) * batch], transform))
        Z = np.concatenate(Z)
        return Z

    def post_gmm_encode(self, X, transform=False, batch=128):
        Z = []
        n_batches = (X.shape[0] + batch - 1) / batch
        for i in range(n_batches):
            x = X[i * batch:(i + 1) * batch]
            z = self.get_child(0).gan.encode(x, transform) if not self.is_leaf else self.gan.encode(x, transform)
            Z.append(z)
        Z = np.concatenate(Z)
        return Z

    def pre_gmm_decode(self, Z):
        return self.gan.decoder(Z)

    def gmm_predict_probs(self, Z):
        priors = [c.prior_prob for c in self.all_child]
        funcs = [multivariate_normal(c.prior_means, c.prior_cov) for c in self.all_child]
        probs = np.array([func.pdf(Z) * prior for prior, func in zip(priors, funcs)]).transpose([1, 0])
        probs = np_utils.prob_dist(probs, axis=-1)
        return probs

    def gmm_predict(self, Z):
        left_dist = np.linalg.norm(self.left.prior_means - Z, axis=-1)
        right_dist = np.linalg.norm(self.right.prior_means - Z, axis=-1)
        preds = np.where(left_dist <= right_dist, 0, 1)
        return preds
        # return np.argmax(self.gmm_predict_probs(Z), axis=-1)

    def gmm_predict_test(self, Z, threshold = 4):

        preds = np.zeros((len(Z)))

        for i in range(len(Z)):
            left_dist = distance.mahalanobis(Z[i], self.kmeans.means[0], self.kmeans.covs[0])
            right_dist = distance.mahalanobis(Z[i], self.kmeans.means[1], self.kmeans.covs[1])
            if left_dist > threshold and right_dist > threshold:
                preds[i] = 2
            elif left_dist < right_dist:
                preds[i] = 0
            else:
                preds[i] = 1
            
        return preds

    def post_gmm_decode(self, Z, train = True, training_list = [], k=0.0, with_PCA = False, threshold = 4):
        if train:
            preds = self.predict_z(as_np((Z)), training_list = training_list)
        else:
            preds = self.gmm_predict_test(as_np(Z), threshold)

        if k > 0.0:
            logger.info(
                colored("Caution!! k is not 0, masking the predictions. better be sure!!", color='yellow', attrs=['bold']))
            probs = self.gmm_predict_probs(as_np(Z))[:, 0]
            indices = np.argsort(probs)[::-1]
            k = int(k * indices.shape[0])
            preds[indices[:k]] = 0
            preds[indices[-k:]] = 1

        if with_PCA:
            if train == False or training_list[0] == 0:
                pcax = PCA(n_components = 42)
                pcax.fit(Z.detach().cpu().numpy())
                self.kmeans.pca = pcax
            
            pcax = self.kmeans.pca
            pcamean = tr.Tensor(pcax.mean_)
            pcaz = tr.Tensor(pcax.components_)
            Z_reduced = tr.matmul(tr.sub(Z, pcamean), tr.transpose(pcaz, 0, 1))
            Z_recon = tr.matmul(Z_reduced, pcaz) + pcamean

        else:
            
            Z_recon = Z

        gan0 = self.get_child(0).gan
        gan1 = self.get_child(1).gan

        # x_mode0 = gan0.decoder.forward(gan0.transform.forward(Z))
        # x_mode1 = gan1.decoder.forward(gan1.transform.forward(Z))

        x_mode0 = gan0.decoder.forward(Z_recon)
        x_mode1 = gan1.decoder.forward(Z_recon)

        X = tr.where(tr.tensor(preds[:, None, None, None]) == 0, x_mode0, x_mode1)

        return X, preds

    # fixed mu

    def init_child_params(self, X, n_components = 2, Z=None, fixed_sigma=True, applyPCA = False):
        if Z is None:
            Z = self.post_gmm_encode(X, transform=False)
        # else:
            # logger.info(colored('Z is not None! Fitting gmm directly on given Z Warning: not calling post_gmm_encoder',
                                # color='red', attrs=['bold']))

        if applyPCA:
        
            pcakmeans = PCA(n_components = 42)
            pcakmeans.fit(Z)
            Z_reduced = pcakmeans.transform(Z)

        else:
            Z_reduced = np.asarray(Z)

        kmeans = KMeans(n_components, max_iter=1000)

        p = kmeans.fit_predict(Z_reduced)

        # means = [None for i in range(n_components)]
        # means1 = [3.0 for i in range(42)]
        # means2 = [-3.0 for i in range(42)]
        # # means1[0] = 3.0
        # # means2[0] = -3.0
        # means = pcakmeans.inverse_transform([means1, means2])

        means1 = [0.8 for i in range(100)]
        means2 = [-0.8 for i in range(100)]
        # means1[0] = 4.5
        # means2[0] = -4.5
        means = np.asarray([means1, means2])

        covs = [None for i in range(n_components)]
        weights = [None for i in range(n_components)]

        for i in range(n_components):
            # print(i)
            Z_temp = Z[np.where(p == i)]
            # print(Z_temp[:32])
            # means[i] = np.mean(Z_temp, axis=0)
            covs[i] = np.eye(Z.shape[-1]) if fixed_sigma else np.cov(Z_temp.T)
            print(len(Z_temp))
            print(len(Z))
            weights[i] = (1.0* len(Z_temp))/len(Z)

        # for i in range(len(p)):
        #     if (distance.mahalanobis(Z[i], means[0], covs[0]) > 2) and (distance.mahalanobis(Z[i], means[1], covs[01]) > 2):
        #         p[i] = 2

        self.kmeans = KMeansCltr(means, covs, weights, p, kmeans.cluster_centers_, pcakmeans)

        # print(means)
        # print(covs)
        print(weights)

    def reassignLabels(self, X, threshold, reassignLabels = False):
        Z = self.post_gmm_encode(X, transform = False)

        preds = self.kmeans.pred

        if reassignLabels:
            for i in range(len(preds)):
                if (distance.mahalanobis(Z[i], self.kmeans.means[0], self.kmeans.covs[0]) > threshold) and (distance.mahalanobis(Z[i], self.kmeans.means[1], self.kmeans.covs[1]) > threshold):
                    preds[i] = 2

        # if not reassignLabels:
        #     for i in range(len(preds)):
        #         if preds[i] == 2:
        #             dis0 = distance.mahalanobis(Z[i], self.kmeans.means[0], self.kmeans.covs[0])
        #             dis1 = distance.mahalanobis(Z[i], self.kmeans.means[1], self.kmeans.covs[1])

        #             if (dis0 < dis1) and (dis0 < threshold):
        #                 preds[i] = 0
        #             elif (dis0 > dis1) and (dis1 < threshold):
        #                 preds[i] = 1

        self.kmeans.pred = preds

    def assignLabels(self, X, percentile, limit, reassignLabels = False):
        Z = self.post_gmm_encode(X, transform = False)

        p = self.kmeans.pred

        if reassignLabels:
            p = np.asarray([2 for i in range(len(p))])

        cluster_distances = []
        cluster_labels = []

        for i in range(len(p)):
            if p[i] == 2:
                dis0 = distance.mahalanobis(Z[i], self.kmeans.means[0], self.kmeans.covs[0])
                dis1 = distance.mahalanobis(Z[i], self.kmeans.means[1], self.kmeans.covs[1])

                # dis0 = distance.mahalanobis(np.dot(Z[i], self.kmeans.means[1] - self.kmeans.means[0]), self.kmeans.means[0], self.kmeans.covs[0])
                # dis1 = distance.mahalanobis(np.dot(Z[i], self.kmeans.means[1] - self.kmeans.means[0]), self.kmeans.means[1], self.kmeans.covs[1])


                d = min(dis0/dis1, dis1/dis0)
                cluster_distances.append(d)

                if d == dis0/dis1:
                    cluster_labels.append(0)
                elif d == dis1/dis0:
                    cluster_labels.append(1)

        sorted_list = np.argsort(cluster_distances)

        print(len(sorted_list))

        total_influx = int(percentile * len(sorted_list))

        print(total_influx)

        if total_influx < limit:
            total_influx = min(limit, len(sorted_list))
            update_list = sorted_list[:total_influx]
        else:
            update_list = sorted_list[:total_influx]

        print(len(update_list))
        print(len(cluster_labels))

        print(update_list)

        # update_list_1 = [int(i) for i in update_list]
        # update_labels = []

        for i in range(len(update_list)):
            p[update_list[i]] = cluster_labels[update_list[i]]
            # update_labels.append(cluster_labels[update_list[i]])

        # p[update_list_1] = update_labels

        self.kmeans.pred = p


    # not fixed mu

    # def init_child_params(self, X, n_components = 2, Z=None, fixed_sigma=True, applyPCA = True):
    #     if Z is None:
    #         Z = self.post_gmm_encode(X, transform=False)
    #     # else:
    #         # logger.info(colored('Z is not None! Fitting gmm directly on given Z Warning: not calling post_gmm_encoder',
    #                             # color='red', attrs=['bold']))

    #     if applyPCA:
        
    #         pcakmeans = PCA(n_components = 5)
    #         pcakmeans.fit(Z)
    #         Z_reduced = pcakmeans.transform(Z)
        
    #     else:
    #         Z_reduced = np.asarray(Z)

    #     kmeans = KMeans(n_components, max_iter=1000)

    #     p = kmeans.fit_predict(Z_reduced)
    #     # print(p[:32])

    #     means = [None for i in range(n_components)]
    #     covs = [None for i in range(n_components)]
    #     weights = [None for i in range(n_components)]

    #     for i in range(n_components):
    #         # print(i)
    #         Z_temp = Z[np.where(p == i)]
    #         # print(Z_temp[:32])
    #         means[i] = np.mean(Z_temp, axis=0)
    #         covs[i] = np.eye(Z.shape[-1]) if fixed_sigma else np.cov(Z_temp.T)
    #         print(len(Z_temp))
    #         print(len(Z))
    #         weights[i] = (1.0* len(Z_temp))/len(Z)

    #     self.kmeans = KMeansCltr(means, covs, weights, p, kmeans.cluster_centers_)

    #     # print(means)
    #     # print(covs)
    #     print(weights)

    def update_child_params(self, X, Z=None, max_iter = 20, fixed_sigma=True, applyPCA = True):
        if Z is None:
            Z = self.post_gmm_encode(X, transform=False)
        # else:
            # logger.info(colored('Z is not None! Fitting gmm directly on given Z Warning: not calling post_gmm_encoder',
                                # color='red', attrs=['bold']))

        if applyPCA:

            pcakmeans = PCA(n_components = 42)
            pcakmeans.fit(Z)
            Z_reduced = pcakmeans.transform(Z)

        else:

            Z_reduced = np.asarray(Z)

        # initialize from previous means

        kmeans = KMeans(self.n_child, init = np.array(self.kmeans.cluster_centers), max_iter=max_iter)


        # initialize new kmeans
        # kmeans = KMeans(self.n_child, max_iter=max_iter)
        p = kmeans.fit_predict(Z_reduced)

        # print(len(p))
        # print(type(p))
        self.kmeans.pred = p

        means = [None for i in range(self.n_child)]
        covs = [None for i in range(self.n_child)]
        weights = [None for i in range(self.n_child)]

        for i in range(self.n_child):
            Z_temp = Z[np.where(p == i)]
            means[i] = np.mean(Z_temp, axis=0)
            covs[i] = np.eye(Z.shape[-1]) if fixed_sigma else np.cov(Z_temp.T)
            # print(len(Z_temp))
            weights[i] = (1.0* len(Z_temp))/len(Z)

        print(weights)

        similar_dist = np.linalg.norm(means[0] - self.kmeans.means[0]) + np.linalg.norm(means[1] - self.kmeans.means[1])
        cross_dist = np.linalg.norm(means[1] - self.kmeans.means[0]) + np.linalg.norm(means[0] - self.kmeans.means[1])

        print(similar_dist-cross_dist)
        # print(cross_dist)
        # print(means[0])
        # print(means[1])
        if similar_dist < cross_dist:
            print("sim")
            for i in range(self.n_child):
                self.get_child(i).update_dist_params(means[i], covs[i], weights[i])
                self.kmeans.means[i] = means[i]
                self.kmeans.covs[i] = covs[i]
                self.kmeans.weights[i] = weights[i]
                self.kmeans.cluster_centers[i] = kmeans.cluster_centers_[i]
        else:
            print("diff")
            for i in range(self.n_child):
                self.get_child(i).update_dist_params(means[1-i], covs[1-i], weights[1-i])
                self.kmeans.means[i] = means[1-i]
                self.kmeans.covs[i] = covs[1-i]
                self.kmeans.weights[i] = weights[1-i]
                self.kmeans.pred = 1 - p
                self.kmeans.cluster_centers[i] = kmeans.cluster_centers_[1-i]

        return (similar_dist - cross_dist)
        # print(self.kmeans.means[0])
        # print(self.kmeans.means[1])

    def fit_gmm(self, X, n_components=2, max_iter=100, warm_start=True, Z=None):
        if self.gmm is None or warm_start == False:
            self.gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, warm_start=False)
        else:
            self.gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, means_init=self.gmm.means_,
                                       precisions_init=self.gmm.precisions_, weights_init=self.gmm.weights_)
        if Z is None:
            Z = self.post_gmm_encode(X, transform=False)
        else:
            logger.info(colored('Z is not None! Fitting gmm directly on given Z Warning: not calling post_gmm_encoder',
                                color='red', attrs=['bold']))
        # print(Z.shape)
        self.gmm.fit(Z)
        for i in range(self.n_child):
            self.get_child(i).update_dist_params(self.gmm.means_[i], self.gmm.covariances_[i], self.gmm.weights_[i])

    def save(self, file):
        pickle_data = {
            'id': self.id,
            'gmm': self.gmm,
            'dist_params': self.dist_params,
            'state_dict': self.state_dict(),
            'name': self.name,
            'kmeans': self.kmeans
        }
        with open(file, 'w') as fp:
            pickle.dump(pickle_data, fp)

    @classmethod
    def load(cls, file, gnode=None, Model=None, strict=False):
        with open(file) as fp:
            pickle_dict = pickle.load(fp)

        node_id = pickle_dict['id']
        name = pickle_dict.get('name', '')
        gmm = pickle_dict['gmm']
        node = gnode or GNode(node_id, Model(name, 2, ))
        node.gmm = gmm
        node.load_state_dict(pickle_dict['state_dict'], strict=strict)
        node.dist_params = pickle_dict['dist_params']
        return node

    @classmethod
    def loadwithkmeans(cls, file, gnode=None, Model=None, strict=False):
        with open(file) as fp:
            pickle_dict = pickle.load(fp)

        node_id = pickle_dict['id']
        name = pickle_dict.get('name', '')
        gmm = pickle_dict['gmm']
        kmeans = pickle_dict['kmeans']
        node = gnode or GNode(node_id, Model(name, 2, ))
        node.gmm = gmm
        node.kmeans = kmeans
        node.load_state_dict(pickle_dict['state_dict'], strict=strict)
        node.dist_params = pickle_dict['dist_params']
        return node

    def __assign_parent(self, parent):
        # type: (GNode) -> None
        self.parent = parent
        if parent is not None:
            parent.child_ids.append(self.id)
            parent.child_nodes[self.id] = self

    def set_child_nodes(self, child_nodes):
        # type: (list[GNode]) -> None
        means = []
        cov = []
        weights = []
        for node in child_nodes:
            self.child_nodes[node.id] = node
            self.child_ids.append(node.id)

            means.append(node.prior_means)
            cov.append(node.prior_cov)
            weights.append(node.prior_prob)

        self.gmm.means_ = np.array(means, dtype='float32')
        self.gmm.covariances_ = np.array(cov, dtype='float32')
        self.gmm.weights_ = np.array(weights, dtype='float32')

    def remove_child(self, child_id):
        self.child_ids.remove(child_id)
        del self.child_nodes[child_id]

    def get_trainer(self):
        # type: () -> GanTrainer
        return self.trainer

    def set_trainer(self, dataloader, hyperparams, train_config, msg='', Model=GanTrainer):
        self.trainer = Model(self.gan, dataloader, hyperparams, train_config, tensorboard_msg=msg)

    #  Phase one optimizers classifier
    def set_optimizer(self):
        encoder_params = list(self.post_gmm_encoder.parameters())
        decoders = self.post_gmm_decoders
        decoder_params = list(decoders[0].parameters()) + list(decoders[1].parameters())
        self.opt_xc = optim.Adam(encoder_params + decoder_params)
        self.opt_xun = optim.Adam(encoder_params + decoder_params)
        self.opt_xr = optim.Adam(decoder_params)
        self.opt_xrecon = optim.Adam(encoder_params + decoder_params)

    def train(self, *args, **kwargs):
        self.trainer.train(*args, **kwargs)

    def step_train_em(self, x_batch, N, fixed_sigma=True):
        Z = self.post_gmm_encode(x_batch)
        p = self.gmm_predict(Z)
        for i, c in enumerate(self.all_child):
            z = Z[np.where(p == i)]
            ni = z.shape[0]
            means = np.mean(z, axis=0)
            means = 1.0 / (N[i] + ni) * (N[i] * c.prior_means + ni * means)

            cov = None if fixed_sigma else np.cov(z.T)

            weight = (1.0 * ni) / Z.shape[0]
            weight = 1.0 / (N[i] + ni) * (N[i] * c.prior_prob + ni * weight)
            c.update_dist_params(means, cov=cov, prior_prob=weight)
            N[i] += ni

    def step_train_x_clf(self, x_batch, training_list = [], clip=0.0, w1 = 1.0, w2 = 1.0, w3 = 1.0, w4 = 1.0, use_pre=False, frozenLabels = True, with_PCA = False, threshold = 4):

        #recon_loss
        w1 = w1
        #likelihood_loss
        w2 = w2
        #hinge_loss
        w3 = w3
        #cross_class_loss
        w4 = w4

        # if start_no == 0:
        #     print(w1)
        #     print(w2)
        #     print(w3)
        #     print(w4)

        id1, id2 = self.child_ids

        # print(id1)
        # print(id2)

        node1 = self.child_nodes[id1]
        node2 = self.child_nodes[id2]

        mu1, cov1, w1 = node1.tensor_params
        mu2, cov2, w2 = node2.tensor_params

        tic = time.time()

        z_batch = self.post_gmm_encoder.forward(x_batch)

        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print(training_list)

        x_recon, preds = self.post_gmm_decode(z_batch, train = True, training_list = training_list, k=0.0, with_PCA = with_PCA, threshold = threshold)

        # print(preds.shape)
        # print(preds)
        # print(training_list)

        if use_pre:
            z_batch_pre = self.pre_gmm_encode(x_batch)
            preds = self.gmm_predict(z_batch_pre)

        if len(z_batch[np.where(preds != 2)]) == 0:
            x_clf_loss_assigned = 0
        else:
            x_clf_loss_assigned = tr.max(tr.tensor(clip), losses.x_clf_loss_assigned(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds != 2)], preds[np.where(preds != 2)]))
        
        if len(z_batch[np.where(preds == 2)]) == 0:
            x_clf_loss_unassigned = 0
        else:
            # x_clf_loss_unassigned = tr.max(tr.tensor(clip), losses.x_clf_loss_unassigned(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds == 2)], preds[np.where(preds == 2)]))
            x_clf_loss_unassigned = losses.x_clf_loss_unassigned(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds == 2)], preds[np.where(preds == 2)])

        x_clf_cross_loss = tr.max(tr.tensor(clip), losses.x_clf_cross_loss(mu1, cov1, w1, mu2, cov2, w2, z_batch, preds))

        tac = time.time()

        batch_size = x_recon.shape[0]

        x_loss_vector = tr.sum((x_recon.view([batch_size, -1]) - x_batch.view([batch_size, -1])) ** 2, dim=-1)

        c = Counter([a for a in preds])
        # weights = tr.Tensor([
        #     1.0 / np.maximum(c[0], 1e-9),
        #     1.0 / np.maximum(c[1], 1e-9),
        #     1.0 / np.maximum(c[2], 1e-9)
        # ])[preds]

        weights_unassigned = tr.Tensor([
            0,
            0,
            1.0 / np.maximum(c[2], 1e-9)
        ])[preds]

        weights_assigned = tr.Tensor([
            1.0 / np.maximum(c[0], 1e-9),
            1.0 / np.maximum(c[1], 1e-9),
            0
        ])[preds]

        # if training_list[0] % 19200 == 0:
        #     print(len(preds))
        #     print(c[0])
        #     print(c[1])
        #     print(c[2])

        x_unassigned_recon_loss = tr.sum(x_loss_vector * weights_unassigned)

        x_assigned_recon_loss = tr.sum(x_loss_vector * weights_assigned)

        # x_recon_loss = tr.sum(x_loss_vector * weights)
        # x_recon_loss = x_unassigned_recon_loss + x_assigned_recon_loss

        loss_assigned = x_assigned_recon_loss + x_clf_loss_assigned
        loss_unassigned = x_unassigned_recon_loss + x_clf_loss_unassigned

        loss_recon = x_assigned_recon_loss + x_unassigned_recon_loss

        # _, cov = tr_utils.mu_cov(z_batch)
        #
        # _, sigmas, _ = tr.svd(cov)
        #
        # sig_sv1 = tr.clamp(tr.svd(cov1)[1], 1e-5) ** 2
        # sig_sv2 = tr.clamp(tr.svd(cov2)[1], 1e-5) ** 2
        #
        # loss_cov1 = tr.sum(sig_sv1)
        # loss_cov2 = tr.sum(sig_sv2)
        # loss_inv_cov1 = tr.sum(1 / sig_sv1)
        # loss_inv_cov2 = tr.sum(1 / sig_sv2)

        # loss = x_recon_loss + 500.0 * x_clf_loss  # + 1e-4 * (loss_inv_cov1 + loss_inv_cov2 + loss_cov1 + loss_cov2)

        # loss = w1 * x_recon_loss + w2 * x_clf_loss + w3 * mu_hinge_loss + w4 * x_clf_cross_loss

        loss = loss_recon + x_clf_loss_assigned + x_clf_loss_unassigned

        # self.opt_xrecon.zero_grad()
        # loss_recon.backward(retain_graph = True)
        # self.opt_xrecon.step()

        self.opt_xc.zero_grad()
        loss_assigned.backward(retain_graph=True)
        self.opt_xc.step()

        self.opt_xun.zero_grad()
        loss_unassigned.backward(retain_graph = True)
        self.opt_xun.step()

        # self.opt_xun.zero_grad()
        # loss.backward(retain_graph = True)
        # self.opt_xun.step()

        if len(z_batch[np.where(preds != 2)]) == 0:
            loss_log_assigned_ch0, loss_log_assigned_ch1 = 0.0, 0.0
        else:       
            loss_log_assigned_ch0, loss_log_assigned_ch1 = losses.x_clf_loss_assigned_separate(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds != 2)], preds[np.where(preds != 2)])
        
        time_taken = tac - tic

        

        return z_batch, x_recon, preds, x_clf_loss_assigned, x_assigned_recon_loss, loss_assigned, x_clf_loss_unassigned, x_unassigned_recon_loss, loss_unassigned, x_clf_cross_loss, loss_recon, loss_log_assigned_ch0, loss_log_assigned_ch1

    def updatePredictions(self, x_batch, training_list, threshold):
        z_batch_update = self.post_gmm_encoder.forward(x_batch)

        x_recon, preds_update = self.post_gmm_decode(z_batch_update, train = True, training_list = training_list, k=0.0, with_PCA = False)

        for i in range(len(preds_update)):
            dis0 = distance.mahalanobis(z_batch_update.detach().cpu().numpy()[i], self.kmeans.means[0], self.kmeans.covs[0])
            dis1 = distance.mahalanobis(z_batch_update.detach().cpu().numpy()[i], self.kmeans.means[1], self.kmeans.covs[1])

            if preds_update[i] == 2:
                if (dis0 < dis1) and (dis0 < threshold):
                # if (dis0 < dis1) and (dis0 < 5.0):
                    preds_update[i] = 0
                elif (dis0 > dis1) and (dis1 < threshold):
                # elif (dis0 > dis1) and (dis1 < 1.5):
                    preds_update[i] = 1

        self.kmeans.pred[training_list] = preds_update

    def step_predict_test(self, x_batch, clip=0.0, use_pre=False, with_PCA = False, threshold = 4):
        
        with tr.no_grad():
            id1, id2 = self.child_ids

            node1 = self.child_nodes[id1]
            node2 = self.child_nodes[id2]

            mu1, cov1, w1 = node1.tensor_params
            mu2, cov2, w2 = node2.tensor_params

            z_batch = self.post_gmm_encoder.forward(x_batch)

            count1 = 0
            count2 = 0

            # for i in range(len(z_batch)):
            #     if distance.mahalanobis(z_batch.detach().cpu().numpy()[0], mu1, cov1) < threshold:
            #         count1 += 1

            #     if distance.mahalanobis(z_batch.detach().cpu().numpy()[0], mu2, cov2) < threshold:
            #         count2 += 1

                # print('xx')
                # print(distance.mahalanobis(z_batch.detach().cpu().numpy()[0], mu1, cov1))
                # print(distance.mahalanobis(z_batch.detach().cpu().numpy()[0], mu2, cov2))
                # print(distance.mahalanobis(mu1, mu2, cov2))
                # print(distance.mahalanobis(mu2, mu1, cov1))
                # print('xx')
            # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            # print(count1)
            # print(count2)
            # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

            x_recon, preds = self.post_gmm_decode(z_batch, train = False, k=0.0, with_PCA = with_PCA, threshold = threshold)

            # print(preds.shape)
            # print(preds)

            if use_pre:
                z_batch_pre = self.pre_gmm_encode(x_batch)
                preds = self.gmm_predict(z_batch_pre)

            if len(z_batch[np.where(preds != 2)]) == 0:
                x_clf_loss_assigned = 0
            else:
                x_clf_loss_assigned = tr.max(tr.tensor(clip), losses.x_clf_loss_assigned(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds != 2)], preds[np.where(preds != 2)]))
            
            if len(z_batch[np.where(preds == 2)]) == 0:
                x_clf_loss_unassigned = 0
            else:
                # x_clf_loss_unassigned = tr.max(tr.tensor(clip), losses.x_clf_loss_unassigned(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds == 2)], preds[np.where(preds == 2)]))
                x_clf_loss_unassigned = losses.x_clf_loss_unassigned(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds == 2)], preds[np.where(preds == 2)])

            x_clf_cross_loss = tr.max(tr.tensor(clip), losses.x_clf_cross_loss(mu1, cov1, w1, mu2, cov2, w2, z_batch, preds))

            batch_size = x_recon.shape[0]

            x_loss_vector = tr.sum((x_recon.view([batch_size, -1]) - x_batch.view([batch_size, -1])) ** 2, dim=-1)

            c = Counter([a for a in preds])
            weights = tr.Tensor([
                1.0 / np.maximum(c[0], 1e-9),
                1.0 / np.maximum(c[1], 1e-9),
                1.0 / np.maximum(c[2], 1e-9)
            ])[preds]

            weights_unassigned = tr.Tensor([
                0,
                0,
                1.0 / np.maximum(c[2], 1e-9)
            ])[preds]

            weights_assigned = tr.Tensor([
                1.0 / np.maximum(c[0], 1e-9),
                1.0 / np.maximum(c[1], 1e-9),
                0
            ])[preds]

            x_unassigned_recon_loss = tr.sum(x_loss_vector * weights_unassigned)

            x_assigned_recon_loss = tr.sum(x_loss_vector * weights_assigned)

            loss_recon = tr.sum(x_loss_vector * weights)

            loss_assigned = x_assigned_recon_loss + x_clf_loss_assigned
            loss_unassigned = x_unassigned_recon_loss + x_clf_loss_unassigned

            if len(z_batch[np.where(preds != 2)]) == 0:
                loss_log_assigned_ch0, loss_log_assigned_ch1 = 0.0, 0.0
            else:       
                loss_log_assigned_ch0, loss_log_assigned_ch1 = losses.x_clf_loss_assigned_separate(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds != 2)], preds[np.where(preds != 2)])


            # loss_recon = x_assigned_recon_loss + x_unassigned_recon_loss

        return preds, x_clf_loss_assigned, x_assigned_recon_loss, loss_assigned, x_clf_loss_unassigned, x_unassigned_recon_loss, loss_unassigned, x_clf_cross_loss, loss_recon, loss_log_assigned_ch0, loss_log_assigned_ch1

    def step_train_mode_separation_nt(self, x_batch, clip=0.0):
        k = self.n_child
        # for i, child in range(k):

        id1, id2 = self.child_ids

        node1 = self.child_nodes[id1]
        node2 = self.child_nodes[id2]

        mu1, cov1, w1 = node1.tensor_params
        mu2, cov2, w2 = node2.tensor_params

        tic = time.time()

        z_batch = self.post_gmm_encoder.forward(x_batch)

        x_recon, preds = self.post_gmm_decode(z_batch, k=0.1)

        x_clf_loss = tr.max(tr.tensor(clip), losses.x_clf_loss(mu1, cov1, w1, mu2, cov2, w2, z_batch))

        tac = time.time()

        batch_size = x_recon.shape[0]

        x_loss_vector = tr.sum((x_recon.view([batch_size, -1]) - x_batch.view([batch_size, -1])) ** 2, dim=-1)

        c = Counter([a for a in preds])
        weights = tr.Tensor([
            1.0 / np.maximum(c[0], 1e-9),
            1.0 / np.maximum(c[1], 1e-9)
        ])[preds]

        x_recon_loss = tr.sum(x_loss_vector * weights)

        _, cov = tr_utils.mu_cov(z_batch)
        #
        _, sigmas, _ = tr.svd(cov)
        #
        sig_sv1 = tr.clamp(tr.svd(cov1)[1], 1e-5) ** 2
        sig_sv2 = tr.clamp(tr.svd(cov2)[1], 1e-5) ** 2
        #
        loss_cov1 = tr.sum(sig_sv1)
        loss_cov2 = tr.sum(sig_sv2)
        loss_inv_cov1 = tr.sum(1 / sig_sv1)
        loss_inv_cov2 = tr.sum(1 / sig_sv2)

        loss = x_recon_loss + 500.0 * x_clf_loss + 1e-4 * (loss_inv_cov1 + loss_inv_cov2 + loss_cov1 + loss_cov2)

        self.opt_xc.zero_grad()
        loss.backward(retain_graph=True)
        self.opt_xc.step()

        time_taken = tac - tic

        return z_batch, x_recon, x_recon_loss, x_clf_loss, loss, preds, time_taken

    def step_train_x_clf_fixed(self, x_batch_left, x_batch_right, clip=0.0):

        mu1, cov1, w1 = self.left.tensor_params
        mu2, cov2, w2 = self.right.tensor_params

        z_batch_left = self.post_gmm_encoder.forward(x_batch_left)
        z_batch_right = self.post_gmm_encoder.forward(x_batch_right)

        x_recon_left = self.left.gan.decoder(z_batch_left)
        x_recon_right = self.right.gan.decoder(z_batch_right)

        x_clf_loss = tr.max(tr.tensor(clip), losses.x_clf_loss_fixed(mu1, cov1, w1, mu2, cov2, w2, z_batch_left, z_batch_right))

        x_recon_loss_left = tr.sum((x_recon_left - x_batch_left) ** 2, dim=-1).mean()
        x_recon_loss_right = tr.sum((x_recon_right - x_batch_right) ** 2, dim=-1).mean()
        x_recon_loss = x_recon_loss_left + x_recon_loss_right

        # _, cov_left = tr_utils.mu_cov(z_batch_left)
        # _, sigmas_left, _ = tr.svd(cov_left)
        #
        # _, cov_right = tr_utils.mu_cov(z_batch_right)
        # _, sigmas_right, _ = tr.svd(cov_right)

        # Don't need regularization, since mu, cov are freezed
        loss = x_recon_loss + x_clf_loss  # + 1e-5 * tr.sum(sigmas_left ** 2)



        self.opt_xc.zero_grad()
        loss.backward(retain_graph=True)
        self.opt_xc.step()

        return x_recon_loss, x_clf_loss, loss

    def set_train_flag(self, mode):
        super(GNode, self).train(mode)

    def pdf(self, x):
        f = stats.multivariate_normal(self.prior_means, cov=self.prior_cov)
        return f.pdf(x)

    def mean_likelihood(self, X):
        Z = self.gan.encode(X)
        return np.mean(self.pdf(Z))

    def sample_z_batch(self, n_samples=1):
        array = np.random.multivariate_normal(self.prior_means, self.prior_cov, n_samples)
        return tr.Tensor(array)

    def sample_restricted_z_batch(self, n_samples=1, prob_threshold=0.02):
        batch = self.sample_z_batch(n_samples * 10)
        batch = self.filter_z(batch, prob_threshold)
        batches, _ = self.split_z(batch)
        reduce = lambda batch: np_utils.random_select(batch, n_samples) if len(batch) > n_samples else batch
        # batches = map(reduce, map(lambda i: batches[i], self.child_ids))
        return {k: reduce(batch) for k, batch in batches.items()}

    def sample_x_batch(self, n_samples=1):
        z_batch = self.sample_z_batch(n_samples)
        return self.gan.decode(x)

    def predict_z(self, Z, training_list = [], probs=False):
        if Z.shape[0] == 0:
            return np.array([])
        if probs:
            if self.gmm:
                P = self.gmm.predict_proba(Z)
            else:
                P = self.kmeans.pred[training_list]
            return P

        if self.gmm:
            Y = self.gmm.predict(Z)
        else:
            Y = self.kmeans.pred[training_list]
        # Y = np.where(Z[:, 0] + Z[:, 1] >= 0, 0, 1)
        # Y = np.array([self.child_ids[y] for y in Y])
        return Y

    def predict_x(self, X, probs=False):
        if X.shape[0] == 0:
            return np.array([])
        Z = self.gan.encode(X)
        return self.predict_z(Z, probs)

    def split_z(self, Z):
        """
        :param Z: np.ndarray of shape [B, F]
        :return:

        z_splits: {
            label : np.ndarray of shape [Bl, F]
        }

        """
        Y = self.predict_z(Z, end_no = Z.shape[0])+1
        labels = self.child_ids
        R = np.arange(Z.shape[0])
        z_splits = {l: Z[np.where(Y == l)] for l in labels}
        i_splits = {l: R[np.where(Y == l)] for l in labels}
        return z_splits, i_splits

    def encoder_helper(self, X):
        samples = X.shape[0]
        iter = (samples // 256) + 1
        z = tr.tensor([])
        for idx in range(iter):
            if idx < iter - 1:
                tempz = tr.from_numpy(self.post_gmm_encode(X[(idx) * 256:(idx + 1) * 256])).cuda()
            else:
                tempz = tr.from_numpy(self.post_gmm_encode(X[(idx) * 256:])).cuda()
            z = tr.cat((z, tempz), 0)

        return z

    def split_x(self, X, Z_flag=False):

        Z = self.post_gmm_encode(X) if not Z_flag else self.encoder_helper(X)
        # print(len(Z))
        z_splits, i_splits = self.split_z(Z)
        x_splits = {l: X[i_split] for l, i_split in i_splits.items()}
        return x_splits, i_splits

    def filter_i(self, Z, prob_threshold):
        probs = self.predict_z(Z, probs=True)
        indices = np.where(probs.min(axis=-1) < prob_threshold)
        return indices

    def filter_z(self, Z, prob_threshold):
        indices = self.filter_i(Z, prob_threshold)
        return Z[indices]

    def filter_x(self, X, prob_threshold):
        Z = self.post_gmm_encode(X)
        indices = self.filter_i(Z, prob_threshold)
        return X[indices]
