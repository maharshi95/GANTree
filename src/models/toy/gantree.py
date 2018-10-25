import logging
from types import NoneType

import numpy as np
from torch import nn
from scipy import stats
from sklearn.mixture import GaussianMixture
from collections import namedtuple

from base.model import BaseGan
from base.model import BaseModel
from trainers.gan_trainer import GanTrainer
from . import nets

logger = logging.getLogger(__name__)

DistParams = namedtuple('Params', 'prior_means prior_cov prior_prob prob')


class GNode(nn.Module):
    def __init__(self, node_id=-1, model=None, parent=None):
        super(GNode, self).__init__()
        self.gan = model
        self.child_ids = []
        self.child_nodes = {}
        self.id = node_id
        self.parent = parent
        self.gmm = GaussianMixture(n_components=2, max_iter=1000)

        self.prior_means = 0.
        self.prior_cov = 0.
        self.prior_prob = 0.
        self.prob = 0.

    def __repr__(self):
        return '<GNode[name={} id={} parent_id={}]>'.format(self.name, self.id, self.parent_id)

    @property
    def name(self):
        return self.gan.name

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
    def child(self):
        return [self.child_nodes[child_id] for child_id in self.child_ids]

    @property
    def parent_name(self):
        return 'nullNode' if self.parent is None else self.parent.name

    @property
    def dist_params(self):
        return self.prior_means, self.prior_cov, self.prior_prob, self.prob

    @property
    def post_gmm_encoder(self):
        return self.child[0].encoder if not self.is_leaf else None

    @dist_params.setter
    def dist_params(self, value):
        self.prior_mean, self.prior_cov, self.prior_prob, self.prob = value

    @property
    def cluster_probs(self):
        return self.gmm.weights_

    def train_gan(self, trainer):
        # type: (GanTrainer) -> None
        trainer.model = self.gan
        trainer.train()

    def get_child(self, child_node_id):
        # type: (int) -> GNode
        return self.child_nodes[child_node_id]

    def pdf(self, x):
        f = stats.multivariate_normal(self.prior_mean, cov=self.prior_cov)
        return f.pdf(x)

    def sample_z_batch(self, n_samples=1):
        return np.random.multivariate_normal(self.prior_mean, self.prior_cov, n_samples)

    def predict_z(self, Z, probs=False):
        if Z.shape[0] == 0:
            return np.array([])
        if probs:
            P = self.gmm.predict_proba(Z)
            return P
        Y = self.gmm.predict(Z)
        Y = np.array([self.child_ids[y] for y in Y])
        return Y

    def predict_x(self, X, probs=False):
        if X.shape[0] == 0:
            return np.array([])
        Z = self.model.encode(X)
        return self.predict_z(Z, probs)

    def mean_likelihood(self, X):
        Z = self.model.encode(X)
        return np.mean(self.pdf(Z))


class GanTree(BaseModel):
    GanModel = None  # type: BaseGan

    def __init__(self, name, GanModel, hyperparams, x_batch=None, n_child=2):
        super(GanTree, self).__init__()
        self.name = name
        self.GanModel = GanModel
        self.H = hyperparams
        self.root_id = 0
        self.nodes = nn.ModuleList()  # type: list[GNode]
        self.n_child = n_child
        self.split_history = []

    @property
    def max_generators(self):
        return len(self.split_history) + 1

    @property
    def n_active_nodes(self):
        return len(self.split_history) + 1

    @property
    def root(self):
        return self.nodes[0]

    def split_models(self, model):
        # type: (BaseGan) -> tuple[BaseGan, BaseGan]
        encoder = model.encoder.copy()

        decoder1 = model.decoder.copy()
        decoder2 = model.decoder.copy()

        disc1 = model.disc.copy()
        disc2 = model.disc.copy()

        model1 = self.GanModel('', encoder, decoder1, disc1)
        model2 = self.GanModel('', encoder, decoder2, disc2)

        return model1, model2

    def create_child_node(self, dist_params, model, parent=None):
        # type: (DistParams, BaseGan, GNode or NoneType) -> GNode
        new_node_id = len(self.nodes)

        model_name = "node-%d" % new_node_id

        model.name = model_name

        logger.info('Creating Child Node: %s' % (model_name))
        logger.info('Node parameters: ')
        logger.info('prior_means: {}'.format(dist_params.prior_means))
        logger.info('prior_cov  : {}'.format(dist_params.prior_cov))
        logger.info('cond_prob  : {}'.format(dist_params.prior_prob))
        logger.info('abs_prob   : {}'.format(dist_params.prob))
        print('')

        new_node = GNode(new_node_id, model, parent)
        new_node.dist_params = dist_params
        self.nodes.append(new_node)

        return new_node

    def split_node(self, parent, x_batch=None):
        x_batch = x_batch or self.x_batch
        assert self.x_batch is not None
        assert isinstance(parent, GNode) or (isinstance(parent, int) and parent < len(self.nodes))
        if isinstance(parent, GNode):
            assert parent.id not in self.split_history
        else:
            assert parent not in self.split_history
            parent = self.nodes[parent]

        logger.info('Starting Split Process: %s' % parent)
        parent.gmm = GaussianMixture(n_components=self.n_child, max_iter=1000)
        z_batch = parent.gan.encode(x_batch)
        parent.gmm.fit(z_batch)
        logger.info('Gaussian Mixture Fitted')
        print('')

        self.mixture_models[parent.id] = parent.gmm
        self.split_history.append(parent.id)

        child_nodes = []

        gan_models = self.split_models(parent.gan)

        for i_child in range(self.n_child):
            n = self.H.z_size
            means = np.ones(n) * 3.0 * np.power(-1, i_child)
            cov = np.eye(n)
            cond_prob = parent.cluster_probs[i_child]
            prob = parent.prob * cond_prob
            child_node_params = DistParams(means, cov, cond_prob, prob)
            child_node = self.create_child_node(child_node_params, gan_models[i_child], parent)
            child_nodes.append(child_node)
        return child_nodes
