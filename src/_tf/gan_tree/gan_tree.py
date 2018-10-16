from collections import deque

import numpy as np
from scipy import stats
import tensorflow as tf
from sklearn import mixture
from exp_context import ExperimentContext
from _tf.models_tf import BaseModel


class GNode(object):
    """
    A single Node of the GANTree which contains the model and a gmm with its parameters,
    along with references to its parent and children nodes.
    """
    gmm = None  # type: mixture.GaussianMixture
    model = None  # type: BaseModel

    def __init__(self, node_id=-1, model=None, parent=None):
        self.model = model
        self.cond_prob = 0.
        self.prob = 0.
        self.means = None
        self.cov = None
        self.child_nodes = {}
        self.child = []
        self.node_id = node_id
        self.parent = parent
        self.gmm = None

    def __repr__(self):
        return '<GNode[name={} node_id={} parent_id={}]>'.format(self.name, self.node_id, self.parent_id)

    @property
    def parent_id(self):
        return -1 if self.parent is None else self.parent.node_id

    @property
    def child_node_ids(self):
        return self.child_nodes.keys()

    @property
    def name(self):
        return self.model.model_name

    @property
    def params(self):
        return self.means, self.cov, self.cond_prob, self.prob

    @params.setter
    def params(self, all_params):
        self.means, self.cov, self.cond_prob, self.prob = all_params

    @property
    def cluster_probs(self):
        return self.gmm.weights_

    def get_child(self, child_node_id):
        return self.child_nodes[child_node_id]

    def pdf(self, x):
        f = stats.multivariate_normal(self.means, cov=self.cov)
        return f.pdf(x)

    def sample_z_batch(self, n_samples=1):
        # TODO: change the impl to gmm.sample
        self.gmm.sample(n_samples)
        return np.random.multivariate_normal(self.means, self.cov, n_samples)

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
        Y = self.predict_z(Z, probs)
        return Y

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
        return np.array([self[i].means for i in range(self.size)])

    @property
    def cov(self):
        return np.array([self[i].cov for i in range(self.size)])

    @property
    def probs(self):
        return np.array([self[i].prob for i in range(self.size)])

    def sample_z_batch(self, n_samples):
        probs = np.array([gan.prob for gan in self.gans])
        gan_ids = np.random.choice(range(self.size), size=n_samples, p=probs)
        z_batch = np.array([self[gan_id].sample_z_batch()[0] for gan_id in gan_ids])
        return z_batch

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
    def __init__(self, name, Model, x_batch, n_child=2):
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
        params = np.zeros(self.H.z_size), np.eye(self.H.z_size), 1.0, 1.0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.Session(config=config)
        self.create_child_node(params, parent=None)
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

    def create_child_node(self, params, parent=None):
        # type: (tuple, {GNode, object}) -> GNode
        new_node_id = len(self.nodes)
        model_name = "%s-%d" % (self.name, new_node_id)
        model = self.Model(model_name, session=self.session)  # type: BaseModel
        model.build()
        model.initiate_service()

        new_node = GNode(new_node_id, model, parent)
        new_node.params = params
        self.nodes.append(new_node)

        if parent is not None:
            model.load_params_from_model(parent.model)
            parent.child.append(new_node)
            parent.child_nodes[new_node_id] = new_node

        return new_node

    def split_node(self, parent):
        assert isinstance(parent, GNode) or (isinstance(parent, int) and parent < len(self.nodes))
        if isinstance(parent, GNode):
            assert parent.node_id not in self.split_history
        else:
            assert parent not in self.split_history
            parent = self.nodes[parent]

        gmm = mixture.GaussianMixture(n_components=self.n_child, covariance_type='full', max_iter=1000)
        parent.gmm = gmm

        z_batch = parent.model.encode(self.x_batch)
        parent.gmm.fit(z_batch)
        self.mixture_models[parent.node_id] = parent.gmm
        self.split_history.append(parent.node_id)

        child_nodes = []

        for i_child in range(self.n_child):
            means = gmm.means_[i_child]
            cov = gmm.covariances_[i_child]
            cond_prob = gmm.weights_[i_child]
            prob = parent.prob * cond_prob
            child_node_params = means, cov, cond_prob, prob
            child_node = self.create_child_node(child_node_params, parent)
            child_nodes.append(child_node)

        return child_nodes

    def get_gans(self, k_clusters):
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
