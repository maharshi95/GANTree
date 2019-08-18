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
from utils import np_utils, tr_utils
from utils.tr_utils import as_np
from .named_tuples import DistParams
from trainers.gan_trainer import GanTrainer
import logging
import math

from configs import Config
from utils.decorators import make_tensor, tensorify

##########  Set Logging  ###########
logger = logging.getLogger(__name__)


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
    kmeans = None # type: KMeansClusters
    child_nodes = None  # type: dict[int, GNode]

    @staticmethod
    def create_clone(node):
        # type: (GNode) -> GNode
        new_node = GNode(node_id=node.id, model=node.gan)
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
    def parent_name(self):
        return 'nullNode' if self.parent is None else self.parent.name

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
        return map(lambda v: tr.tensor(v, dtype=tr.float32).cuda(), [m, s, w])

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
        
        return self.child_nodes[self.child_ids[index]]

    @property
    def pre_gmm_encoder(self):
        return self.gan.encoder

    @property
    def pre_gmm_decoder(self):
        return self.gan.generator

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
        X = X.cuda()
        Z = []
        n_batches = (X.shape[0] + batch - 1) // batch
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

    def post_gmm_decode(self, Z, train = True, training_list = [], with_PCA = False, threshold = 4):
        if train:
            preds = self.predict_z(as_np((Z)), training_list = training_list)
        else:
            preds = self.gmm_predict_test(as_np(Z), threshold)

        if with_PCA:
            if train == False or training_list[0] == 0:
                pcax = PCA(n_components = 3)
                pcax.fit(Z.detach().cpu().numpy())
                self.kmeans.pca = pcax
            
            pcax = self.kmeans.pca
            pcamean = tr.Tensor(pcax.mean_).cuda()
            pcaz = tr.Tensor(pcax.components_).cuda()
            Z_reduced = tr.matmul(tr.sub(Z, pcamean), tr.transpose(pcaz, 0, 1))
            Z_recon = tr.matmul(Z_reduced, pcaz) + pcamean

        else:
            
            Z_recon = Z

        gan0 = self.get_child(0).gan
        gan1 = self.get_child(1).gan

        x_mode0 = gan0.generator.forward(Z_recon)
        x_mode1 = gan1.generator.forward(Z_recon)

        X = tr.where(tr.tensor(preds[:, None, None, None]).cuda() == 0, x_mode0, x_mode1)

        return X, preds


    def init_child_params(self, X, n_components = 2, Z=None, fixed_sigma=True, applyPCA = False, H = None):
        dmu = H.dmu
        value = 0.5 * dmu / math.sqrt(H.z_dim)

        if Z is None:
            Z = self.post_gmm_encode(X, transform=False)
        

        if applyPCA:
        
            pcakmeans = PCA(n_components = 42)
            pcakmeans.fit(Z)
            Z_reduced = pcakmeans.transform(Z)

        else:
            Z_reduced = np.asarray(Z)

        kmeans = KMeans(n_components, max_iter=1000)

        p = kmeans.fit_predict(Z_reduced)


        means1 = [self.prior_means[i] + value for i in range(H.z_dim)]
        means2 = [self.prior_means[i] - value for i in range(H.z_dim)]

        means = np.asarray([means1, means2])

        covs = [None for i in range(n_components)]
        weights = [None for i in range(n_components)]

        for i in range(n_components):
            Z_temp = Z[np.where(p == i)]
            covs[i] = np.eye(Z.shape[-1]) if fixed_sigma else np.cov(Z_temp.T)
            weights[i] = (1.0* len(Z_temp))/len(Z)

        self.kmeans = KMeansCltr(means, covs, weights, p, kmeans.cluster_centers_, None)


    def update_child_params(self, X, Z=None, max_iter = 20, fixed_sigma=True, applyPCA = True):
        if Z is None:
            Z = self.post_gmm_encode(X, transform=False)
        
        if applyPCA:

            pcakmeans = PCA(n_components = 42)
            pcakmeans.fit(Z)
            Z_reduced = pcakmeans.transform(Z)

        else:

            Z_reduced = Z

        # initialize from previous means
        kmeans = KMeans(self.n_child, init = np.array(self.kmeans.cluster_centers), max_iter=max_iter)


        # initialize new kmeans
        # kmeans = KMeans(self.n_child, max_iter=max_iter)

        p = kmeans.fit_predict(Z_reduced)

        print(p.shape)

        self.kmeans.pred = p

        means = [None for i in range(self.n_child)]
        covs = [None for i in range(self.n_child)]
        weights = [None for i in range(self.n_child)]

        for i in range(self.n_child):
            Z_temp = Z[np.where(p == i)]
            means[i] = np.mean(Z_temp, axis=0)
            covs[i] = np.eye(Z.shape[-1]) if fixed_sigma else np.cov(Z_temp.T)
            weights[i] = (1.0* len(Z_temp))/len(Z)

        print(weights)

        similar_dist = np.linalg.norm(means[0] - self.kmeans.means[0]) + np.linalg.norm(means[1] - self.kmeans.means[1])
        cross_dist = np.linalg.norm(means[1] - self.kmeans.means[0]) + np.linalg.norm(means[0] - self.kmeans.means[1])

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


    def save(self, file):
        pickle_data = {
            'id': self.id,
            'dist_params': self.dist_params,
            'state_dict': self.state_dict(),
            'name': self.name,
            'kmeans': self.kmeans
        }
        with open(file, 'wb') as fp:
            pickle.dump(pickle_data, fp)

    @classmethod
    def load(cls, file, gnode=None, Model=None, strict=False):
        with open(file, 'rb') as fp:
            pickle_dict = pickle.load(fp)

        node_id = pickle_dict['id']
        name = pickle_dict.get('name', '')
        kmeans = pickle_dict['kmeans']
        node = gnode or GNode(node_id, Model(name, 2, ))
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
        self.opt_xassigned = optim.Adam(encoder_params + decoder_params)
        self.opt_xunassigned = optim.Adam(encoder_params + decoder_params)
        self.opt_xrecon = optim.Adam(encoder_params + decoder_params)

    def train(self, *args, **kwargs):
        self.trainer.train(*args, **kwargs)

    def step_train_x_clf_phase1(self, x_batch, training_list = [], clip=0.0, frozenLabels = True, with_PCA = False, threshold = 4):

        id1, id2 = self.child_ids

        node1 = self.child_nodes[id1]
        node2 = self.child_nodes[id2]

        mu1, cov1, w1 = node1.tensor_params
        mu2, cov2, w2 = node2.tensor_params

        z_batch = self.post_gmm_encoder.forward(x_batch)

        x_recon, preds = self.post_gmm_decode(z_batch, train = True, training_list = training_list, with_PCA = with_PCA, threshold = threshold)


        if len(z_batch[np.where(preds != 2)]) == 0:
            x_clf_loss_assigned = 0
        else:
            x_clf_loss_assigned = tr.max(tr.tensor(clip).cuda(), losses.x_clf_loss_assigned(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds != 2)], preds[np.where(preds != 2)]))
        

        if len(z_batch[np.where(preds == 2)]) == 0:
            x_clf_loss_unassigned = 0
        else:
            x_clf_loss_unassigned = tr.max(tr.tensor(clip).cuda(), losses.x_clf_loss_unassigned(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds == 2)], preds[np.where(preds == 2)]))


        x_clf_cross_loss = tr.max(tr.tensor(clip).cuda(), losses.x_clf_cross_loss(mu1, cov1, w1, mu2, cov2, w2, z_batch, preds))


        batch_size = x_recon.shape[0]
        x_loss_vector = tr.sum((x_recon.view([batch_size, -1]) - x_batch.view([batch_size, -1])) ** 2, dim=-1)


        c = Counter([a for a in preds])

        weights_unassigned = tr.Tensor([
            0,
            0,
            1.0 / np.maximum(c[2], 1e-9)
        ]).cuda()[preds]

        weights_assigned = tr.Tensor([
            1.0 / np.maximum(c[0], 1e-9),
            1.0 / np.maximum(c[1], 1e-9),
            0
        ]).cuda()[preds]

        

        x_unassigned_recon_loss = 1e-2 * tr.sum(x_loss_vector * weights_unassigned)
        x_assigned_recon_loss = 1e-2 * tr.sum(x_loss_vector * weights_assigned)


        loss_assigned = x_assigned_recon_loss + x_clf_loss_assigned
        loss_unassigned = x_unassigned_recon_loss + x_clf_loss_unassigned


        loss_recon = x_assigned_recon_loss + x_unassigned_recon_loss


        loss = loss_recon + x_clf_loss_assigned + x_clf_loss_unassigned


        self.opt_xassigned.zero_grad()
        loss_assigned.backward(retain_graph=True)
        self.opt_xassigned.step()

        self.opt_xunassigned.zero_grad()
        loss_unassigned.backward(retain_graph = True)
        self.opt_xunassigned.step()

        return z_batch, x_recon, preds, x_clf_loss_assigned, x_assigned_recon_loss, loss_assigned, x_clf_loss_unassigned, x_unassigned_recon_loss, loss_unassigned, x_clf_cross_loss, loss_recon

    def reassignLabels(self, X, threshold):
        Z = self.post_gmm_encode(X, transform = False)

        preds = self.kmeans.pred

        for i in range(len(preds)):
            if (distance.mahalanobis(Z[i], self.kmeans.means[0], self.kmeans.covs[0]) > threshold) and (distance.mahalanobis(Z[i], self.kmeans.means[1], self.kmeans.covs[1]) > threshold):
                preds[i] = 2

        self.kmeans.pred = preds

    def updatePredictions(self, x_batch, training_list, threshold):
        z_batch_update = self.post_gmm_encoder.forward(x_batch)

        x_recon, preds_update = self.post_gmm_decode(z_batch_update, train = True, training_list = training_list, with_PCA = False)

        for i in range(len(preds_update)):
            dis0 = distance.mahalanobis(z_batch_update.detach().cpu().numpy()[i], self.kmeans.means[0], self.kmeans.covs[0])
            dis1 = distance.mahalanobis(z_batch_update.detach().cpu().numpy()[i], self.kmeans.means[1], self.kmeans.covs[1])

            if preds_update[i] == 2:
                if (dis0 < dis1) and (dis0 < threshold):
                    preds_update[i] = 0
                elif (dis0 > dis1) and (dis1 < threshold):
                    preds_update[i] = 1

        self.kmeans.pred[training_list] = preds_update

  
    def step_predict_test(self, x_batch, clip=0.0, with_PCA = False, threshold = 4):
        
        with tr.no_grad():
            id1, id2 = self.child_ids

            node1 = self.child_nodes[id1]
            node2 = self.child_nodes[id2]

            mu1, cov1, w1 = node1.tensor_params
            mu2, cov2, w2 = node2.tensor_params

            z_batch = self.post_gmm_encoder.forward(x_batch)

            x_recon, preds = self.post_gmm_decode(z_batch, train = False, with_PCA = with_PCA, threshold = threshold)


            if len(z_batch[np.where(preds != 2)]) == 0:
                x_clf_loss_assigned = 0
            else:
                x_clf_loss_assigned = tr.max(tr.tensor(clip).cuda(), losses.x_clf_loss_assigned(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds != 2)], preds[np.where(preds != 2)]))
            

            if len(z_batch[np.where(preds == 2)]) == 0:
                x_clf_loss_unassigned = 0
            else:
                x_clf_loss_unassigned = losses.x_clf_loss_unassigned(mu1, cov1, w1, mu2, cov2, w2, z_batch[np.where(preds == 2)], preds[np.where(preds == 2)])


            x_clf_cross_loss = tr.max(tr.tensor(clip).cuda(), losses.x_clf_cross_loss(mu1, cov1, w1, mu2, cov2, w2, z_batch, preds))


            batch_size = x_recon.shape[0]
            x_loss_vector = tr.sum((x_recon.view([batch_size, -1]) - x_batch.view([batch_size, -1])) ** 2, dim=-1)

            c = Counter([a for a in preds])
            weights = tr.Tensor([
                1.0 / np.maximum(c[0], 1e-9),
                1.0 / np.maximum(c[1], 1e-9),
                1.0 / np.maximum(c[2], 1e-9)
            ]).cuda()[preds]

            weights_unassigned = tr.Tensor([
                0,
                0,
                1.0 / np.maximum(c[2], 1e-9)
            ]).cuda()[preds]

            weights_assigned = tr.Tensor([
                1.0 / np.maximum(c[0], 1e-9),
                1.0 / np.maximum(c[1], 1e-9),
                0
            ]).cuda()[preds]

            x_unassigned_recon_loss = 1e-2 * tr.sum(x_loss_vector * weights_unassigned)

            x_assigned_recon_loss = 1e-2 * tr.sum(x_loss_vector * weights_assigned)

            loss_recon = 1e-2 * tr.sum(x_loss_vector * weights)

            loss_assigned = x_assigned_recon_loss + x_clf_loss_assigned
            loss_unassigned = x_unassigned_recon_loss + x_clf_loss_unassigned


        return preds, x_clf_loss_assigned, x_assigned_recon_loss, loss_assigned, x_clf_loss_unassigned, x_unassigned_recon_loss, loss_unassigned, x_clf_cross_loss, loss_recon


    def set_train_flag(self, mode):
        super(GNode, self).train(mode)

    def pdf(self, x):
        f = stats.multivariate_normal(self.prior_means, cov=self.prior_cov)
        return f.pdf(x)

    def mean_likelihood(self, X):
        Z = self.gan.encode(X)
        return np.mean(self.pdf(Z))

    def sample_x_batch(self, n_samples=1):
        z_batch = self.sample_z_batch(n_samples)
        return self.gan.decode(x)

    def predict_z(self, Z, training_list = [], probs=False):
        if Z.shape[0] == 0:
            return np.array([])
        if probs:    
            P = self.kmeans.pred[training_list]
            return P

        
        Y = self.kmeans.pred[training_list]

        return Y

    def predict_x(self, X, probs=False):
        if X.shape[0] == 0:
            return np.array([])
        Z = self.gan.encode(X)
        return self.predict_z(Z, probs)

    def split_z(self, Z):
        Y = self.gmm_predict(Z)
        labels = self.child_ids
        R = np.arange(Z.shape[0])

        z_splits = {labels[l]: Z[np.where(Y == l)] for l in range(len(labels))}
        i_splits = {labels[l]: R[np.where(Y == l)] for l in range(len(labels))}

        return z_splits, i_splits

    def encoder_helper(self, X):
        samples = X.shape[0]
        iter = (samples // 256) + 1
        z = tr.tensor([])
        for idx in range(iter):
            if idx < iter - 1:
                tempz = tr.from_numpy(self.post_gmm_encode(X[(idx) * 256:(idx + 1) * 256])) 
            else:
                tempz = tr.from_numpy(self.post_gmm_encode(X[(idx) * 256:]))

            z = tr.cat((z, tempz), 0)

        return z

    def split_x(self, X, Z_flag=False):
        Z = self.post_gmm_encode(X) if not Z_flag else self.encoder_helper(X)
        _, i_splits = self.split_z(Z)
        return i_splits
