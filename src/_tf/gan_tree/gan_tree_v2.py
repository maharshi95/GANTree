import logging
from collections import deque, namedtuple
from types import NoneType

import numpy as np
from scipy import stats
import tensorflow as tf
from sklearn import mixture

from exp_context import ExperimentContext
from _tf.models_tf import BaseModel

logger = logging.getLogger(__name__)

Params = namedtuple('Params', 'prior_mean prior_cov cond_prob abs_prob')

class GNode(object):
    """
    A single Node of the GANTree which contains the model and a gmm with its parameters,
    along with references to its parent and children nodes.
    """
    child_nodes = None  # type: dict[int, GNode]
    parent = None  # type: GNode
    gmm = None  # type: mixture.GaussianMixture
    model = None  # type: BaseModel

    def __init__(self, node_id=-1, model=None, parent=None):
        self.model = model
        self.cond_prob = 0.
        self.prob = 0.
        self.child_nodes = {}
        self.child = []
        self.node_id = node_id
        self.parent = parent
        self.gmm = None
        self.prior_mean = None
        self.prior_cov = None

    def __repr__(self):
        return '<GNode[name={} node_id={} parent_id={}]>'.format(self.name, self.node_id, self.parent_id)

    @property
    def is_root(self):
        return self.parent is None

    @property
    def id(self):
        return self.node_id

    @property
    def parent_id(self):
        return -1 if self.parent is None else self.parent.node_id

    @property
    def child_node_ids(self):
        return self.child_nodes.keys()

    @property
    def name(self):
        return self.model.name

    @property
    def model_name(self):
        return self.model.name

    @property
    def parent_name(self):
        return 'null_node' if self.parent is None else self.parent.name

    @property
    def model_scope(self):
        return self.model.model_scope

    @property
    def params(self):
        return self.prior_mean, self.prior_cov, self.cond_prob, self.prob

    @params.setter
    def params(self, all_params):
        self.prior_mean, self.prior_cov, self.cond_prob, self.prob = all_params

    @property
    def cluster_probs(self):
        return self.gmm.weights_

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
        Y = np.array([self.child[y].node_id for y in Y])
        return Y

    def predict_x(self, X, probs=False):
        if X.shape[0] == 0:
            return np.array([])
        Z = self.model.encode(X)
        return self.predict_z(Z, probs)

    def mean_likelihood(self, X):
        Z = self.model.encode(X)
        return np.mean(self.pdf(Z))

    def split_z(self, Z):
        """
        :param Z: np.ndarray of shape [B, F]
        :return:

        z_splits: {
            label : np.ndarray of shape [Bl, F]
        }

        """
        Y = self.predict_z(Z)
        labels = [cid for cid in self.child_node_ids]
        R = np.arange(Z.shape[0])
        z_splits = {l: Z[np.where(Y == l)] for l in labels}
        i_splits = {l: R[np.where(Y == l)] for l in labels}
        return z_splits, i_splits

    def split_x(self, X):
        Z = self.model.encode(X)
        z_splits, i_splits = self.split_z(Z)
        x_splits = {l: X[i_split] for l, i_split in i_splits.items()}
        return x_splits, i_splits


class GANSet(object):
    def __init__(self, session, gan_nodes, root):
        self.gans = gan_nodes
        self.session = session
        self.root = root

    def __getitem__(self, item):
        # type: (int) -> GNode
        return self.gans[item]

    def __iter__(self):
        return iter(self.gans)

    def __len__(self):
        # type: () -> int
        return len(self.gans)

    @property
    def size(self):
        return len(self.gans)

    @property
    def means(self):
        return np.array([self[i].prior_mean for i in range(self.size)])

    @property
    def cov(self):
        return np.array([self[i].prior_cov for i in range(self.size)])

    @property
    def probs(self):
        return np.array([self[i].prob for i in range(self.size)])

    def sample_z_batch(self, n_samples):
        probs = np.array([gan.prob for gan in self.gans])
        gan_ids = np.random.choice(range(self.size), size=n_samples, p=probs)
        z_batch = np.array([self[gan_id].sample_z_batch()[0] for gan_id in gan_ids])
        return gan_ids, z_batch

    def predict_z(self, Z):
        n_samples = Z.shape[0]

        labels = np.zeros(n_samples)

        Z_splits = {0: Z}
        I_splits = {0: np.arange(n_samples)}

        fringe_set = deque([self.root])
        while fringe_set:
            gnode = fringe_set.popleft()
            z_splits, i_splits = gnode.split_z(Z_splits[gnode.node_id])
            Z_splits.update(z_splits)
            I_splits.update(i_splits)
            for child in gnode.child:
                if child not in self.gans:
                    fringe_set.append(child)
        for gnode in self.gans:
            cluster_id = gnode.node_id
            indices = I_splits[cluster_id]
            labels[indices] = cluster_id

        return labels, Z_splits

    def predict_x(self, X):
        n_samples = X.shape[0]

        labels = np.zeros(n_samples)

        X_splits = {0: X}
        I_splits = {0: np.arange(n_samples)}

        fringe_set = deque([self.root])
        while fringe_set:
            gnode = fringe_set.popleft()
            x_splits, i_splits = gnode.split_x(X_splits[gnode.node_id])
            X_splits.update(x_splits)
            I_splits.update(i_splits)
            for child in gnode.child:
                if child not in self.gans:
                    fringe_set.append(child)
        for gnode in self.gans:
            cluster_id = gnode.node_id
            indices = I_splits[cluster_id]
            labels[indices] = cluster_id

        return labels, X_splits


class GanTree(object):
    def __init__(self, name, Model, x_batch=None, n_child=2):
        self.name = name
        self.Model = Model
        self.x_batch = x_batch
        self.H = ExperimentContext.Hyperparams
        self.root_idx = -1
        self.nodes = []  # type: list[GNode]
        self.n_child = n_child
        self.split_history = []
        self.mixture_models = {}
        self._is_initiated = False

    def initiate(self):
        assert self._is_initiated == False
        params = Params(np.zeros(self.H.z_size), np.eye(self.H.z_size), 1.0, 1.0)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.Session(config=config)
        self.create_child_node(params, parent=None, initialize_shared_variables=True)
        self._is_initiated = True

    def parent(self, gnode):
        return self.nodes[gnode.parent_id]

    @property
    def max_generators(self):
        return len(self.split_history) + 1

    @property
    def n_active_nodes(self):
        return len(self.split_history) + 1

    @property
    def root(self):
        return self.nodes[0]

    @property
    def scope(self):
        return self.name

    def create_child_node(self, params, parent=None, initialize_shared_variables=False):
        # type: (Params, GNode or NoneType, bool) -> GNode
        new_node_id = len(self.nodes)

        model_name = "node-%d" % new_node_id
        model_scope = "%s/%s" % (self.scope, model_name)

        logger.info('Creating Child Node: %s with scope %s' % (model_name, model_scope))
        logger.info('Node parameters: ')
        logger.info('prior_means: {}'.format(params.prior_mean))
        logger.info('prior_cov  : {}'.format(params.prior_cov))
        logger.info('cond_prob  : {}'.format(params.cond_prob))
        logger.info('abs_prob   : {}'.format(params.abs_prob))
        print('')

        shared_scope = self.shared_scope(parent) if parent else self.default_shared_scope()
        model = self.Model(model_name, session=self.session,
                           shared_scope=shared_scope,
                           model_scope=model_scope,
                           z_mean=params.prior_mean,
                           z_cov=params.prior_cov)  # type: BaseModel
        model.build()
        model.initiate_service(initialize_shared_variables=initialize_shared_variables)

        new_node = GNode(new_node_id, model, parent)
        new_node.params = params
        self.nodes.append(new_node)

        if parent is not None:
            model.load_params_from_model(parent.model)
            parent.child.append(new_node)
            parent.child_nodes[new_node_id] = new_node

        return new_node

    def split_node(self, parent, x_batch=None):
        x_batch = x_batch or self.x_batch
        assert self.x_batch is not None
        assert isinstance(parent, GNode) or (isinstance(parent, int) and parent < len(self.nodes))
        if isinstance(parent, GNode):
            assert parent.node_id not in self.split_history
        else:
            assert parent not in self.split_history
            parent = self.nodes[parent]

        logger.info('Starting Split Process: %s' % parent)
        gmm = mixture.GaussianMixture(n_components=self.n_child, covariance_type='full', max_iter=1000)
        parent.gmm = gmm
        z_batch = parent.model.encode(x_batch)
        parent.gmm.fit(z_batch)
        logger.info('Gaussian Mixture Fitted')
        print('')

        self.mixture_models[parent.node_id] = parent.gmm
        self.split_history.append(parent.node_id)

        child_nodes = []

        initialize_shared_variables = True
        for i_child in range(self.n_child):
            n = self.H.z_size
            means = np.ones(n) * 3.0 * np.power(-1, i_child)
            cov = np.eye(n)
            cond_prob = gmm.weights_[i_child]
            prob = parent.prob * cond_prob
            child_node_params = Params(means, cov, cond_prob, prob)
            child_node = self.create_child_node(child_node_params, parent, initialize_shared_variables)
            initialize_shared_variables = False
            child_nodes.append(child_node)
            print('')
        return child_nodes

    def create_ganset(self, k_clusters):
        active_nodes = {self.nodes[0]}
        for i in range(k_clusters - 1):
            split_node = self.nodes[self.split_history[i]]
            active_nodes.remove(split_node)
            for child_node in split_node.child:
                active_nodes.add(child_node)
        return GANSet(self.session, list(active_nodes), self.root)

    def _recursive_shutdown(self, node_id):
        for child_node_id in self.nodes[node_id].child:
            self._recursive_shutdown(child_node_id)
        model = self.nodes[node_id].model
        model.session.close()

    def shutdown(self):
        self._recursive_shutdown(0)

    def default_shared_scope(self):
        """
        :return: the shared scope name used by the root node.
        """
        return self.scope + '/shared/null_node'

    def shared_scope(self, parent_node=None):
        return self.scope + '/shared/' + parent_node.model_name
