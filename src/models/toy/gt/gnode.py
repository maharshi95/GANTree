from __future__ import absolute_import
import pickle
from collections import Counter

import numpy as np
import torch as tr
from scipy import stats
from sklearn.mixture import GaussianMixture
from torch import nn, optim

from models import losses
from models.toy.gan import ToyGAN
from utils import np_utils, tr_utils, viz_utils
from utils.tr_utils import as_np
from .named_tuples import DistParams
from trainers.gan_trainer import GanTrainer


class GMM(object):
    def __init__(self, means, cov, weights):
        self.means = means
        self.cov = cov
        self.weights = weights


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
    trainer = None  # type: GanTrainer
    opt_xc = None  # type: optim.Adam
    opt_xr = None  # type: optim.Adam
    gmm = None  # type: GaussianMixture
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

    def pre_gmm_encode(self, X, transform=False):
        return self.gan.encode(X, transform)

    def post_gmm_encode(self, X, transform=False):
        return self.get_child(0).gan.encode(X, transform) if not self.is_leaf else self.gan.encode(X, transform)

    def pre_gmm_decode(self, Z):
        return self.gan.decoder(self.gan.transform(Z))

    def post_gmm_decode(self, Z):
        preds = self.gmm.predict(as_np((Z)))

        gan0 = self.get_child(0).gan
        gan1 = self.get_child(1).gan

        # x_mode0 = gan0.decoder.forward(gan0.transform.forward(Z))
        # x_mode1 = gan1.decoder.forward(gan1.transform.forward(Z))

        x_mode0 = gan0.decoder.forward(Z)
        x_mode1 = gan1.decoder.forward(Z)

        X = tr.where(tr.tensor(preds[:, None]) == 0, x_mode0, x_mode1)

        return X, preds

    def fit_gmm(self, X, n_components=2, max_iter=1000, warm_start=True):
        if self.gmm is None or warm_start == False:
            self.gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, warm_start=False)
        else:
            self.gmm.n_components = n_components
            self.gmm.max_iter = max_iter
        Z = self.post_gmm_encode(X, transform=False)
        self.gmm.fit(Z)

        for i in range(self.n_child):
            self.get_child(i).update_dist_params(self.gmm.means_[i], self.gmm.covariances_[i], self.gmm.weights_[i])

    def save(self, file):
        pickle_data = {
            'id': self.id,
            'gmm': self.gmm,
            'dist_params': self.dist_params,
            'state_dict': self.state_dict(),
            'name': self.name
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

    def set_optimizer(self):
        encoder_params = list(self.post_gmm_encoder.parameters())
        decoders = self.post_gmm_decoders
        decoder_params = list(decoders[0].parameters()) + list(decoders[1].parameters())
        self.opt_xc = optim.Adam(encoder_params + decoder_params)
        self.opt_xr = optim.Adam(decoder_params)

    def train(self, *args, **kwargs):
        self.trainer.train(*args, **kwargs)

    def step_train_x_clf(self, x_batch, clip=0.0):
        id1, id2 = self.child_ids

        node1 = self.child_nodes[id1]
        node2 = self.child_nodes[id2]

        mu1, cov1, w1 = node1.tensor_params
        mu2, cov2, w2 = node2.tensor_params

        z_batch = self.post_gmm_encoder.forward(x_batch)

        x_recon, preds = self.post_gmm_decode(z_batch)

        x_clf_loss = tr.max(tr.tensor(clip), losses.x_clf_loss(mu1, cov1, w1, mu2, cov2, w2, z_batch))

        x_loss_vector = tr.sum((x_recon - x_batch) ** 2, dim=-1)
        c = Counter([a for a in preds])
        weights = tr.Tensor([
            1.0 / np.maximum(c[0], 1e-9),
            1.0 / np.maximum(c[1], 1e-9)
        ])[preds]

        x_recon_loss = tr.sum(x_loss_vector * weights)

        _, cov = tr_utils.mu_cov(z_batch)

        _, sigmas, _ = tr.svd(cov)

        loss = x_recon_loss + 100.0 * x_clf_loss + 1e-5 * tr.sum(sigmas ** 2)

        self.opt_xc.zero_grad()
        loss.backward(retain_graph=True)
        self.opt_xc.step()

        return z_batch, x_recon, x_recon_loss, x_clf_loss, loss

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
        return self.gan.decode(z_batch)

    def predict_z(self, Z, probs=False):
        if Z.shape[0] == 0:
            return np.array([])
        if probs:
            P = self.gmm.predict_proba(Z)
            return P
        Y = self.gmm.predict(Z)
        # Y = np.where(Z[:, 0] + Z[:, 1] >= 0, 0, 1)
        Y = np.array([self.child_ids[y] for y in Y])
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
        Y = self.predict_z(Z)
        labels = self.child_ids
        R = np.arange(Z.shape[0])
        z_splits = {l: Z[np.where(Y == l)] for l in labels}
        i_splits = {l: R[np.where(Y == l)] for l in labels}
        return z_splits, i_splits

    def split_x(self, X):
        Z = self.post_gmm_encode(X)
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
