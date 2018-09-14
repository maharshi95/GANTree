import numpy as np
import tensorflow as tf
from sklearn import mixture
from exp_context import ExperimentContext


class GNode(object):
    def __init__(self, node_id=-1, model=None, parent=None):
        self.model = model
        self.cond_prob = 0.
        self.prob = 0.
        self.means = None
        self.cov = None
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
    def child_ids(self):
        return [c.node_id for c in self.child]

    @property
    def name(self):
        return self.model.model_name

    @property
    def params(self):
        return self.means, self.cov, self.cond_prob, self.prob

    @params.setter
    def params(self, all_params):
        self.means, self.cov, self.cond_prob, self.prob = all_params

    def sample(self, n_samples=1):
        return np.random.multivariate_normal(self.means, self.cov, n_samples)

    def predict(self, X):
        Y = self.gmm.predict(X)
        Y = np.array([self.child[y].node_id for y in Y])
        return Y

    def split(self, X):
        Y = self.predict(X)
        labels = [cid for cid in self.child_ids]
        x_splits = np.array([X[np.where(Y == cid)] for cid in self.child_ids])
        i_splits = np.array([np.arange(X.shape[0])[np.where(Y == cid)] for cid in self.child_ids])
        return labels, x_splits, i_splits


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
        z_batch = np.array([self[gan_id].sample()[0] for gan_id in gan_ids])
        return z_batch

    def predict(self, X):
        n_samples = X.shape[0]

        labels = np.zeros(n_samples)

        X_splits = {0: X}
        I_splits = {0: np.arange(n_samples)}

        fringe_set = {self.root}
        while fringe_set:
            for gnode in fringe_set:
                fringe_set.remove(gnode)
                l, x_splits, i_splits = gnode.split(X_splits[gnode.node_id])
                for i in range(len(x_splits)):
                    X_splits[l[i]] = x_splits[i]
                    I_splits[l[i]] = i_splits[i]
                for child in gnode.child:
                    if child not in self.gans:
                        fringe_set.add(child)
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
        self.add_node(params, parent=None)
        self._is_initiated = True

    def parent(self, gnode):
        return self.nodes[gnode.parent_id]

    @property
    def max_generators(self):
        return len(self.split_history) + 1

    def add_node(self, params, parent=None):
        new_node_id = len(self.nodes)
        model_name = "%s-%d" % (self.name, new_node_id)
        model = self.Model(model_name, session=self.session)
        model.build()
        model.initiate_service()

        new_node = GNode(new_node_id, model, parent)
        new_node.params = params
        self.nodes.append(new_node)

        if parent is not None:
            model.load_params_from_model(parent.model)
            parent.child.append(new_node)

    def split(self, parent):
        assert isinstance(parent, GNode)
        assert parent.node_id not in self.split_history
        gmm = mixture.GaussianMixture(n_components=self.n_child, covariance_type='full', max_iter=1000)
        parent.gmm = gmm

        z_batch = parent.model.encode(self.x_batch)
        parent.gmm.fit(z_batch)
        self.mixture_models[parent.node_id] = parent.gmm
        self.split_history.append(parent.node_id)

        for i_child in range(self.n_child):
            means = gmm.means_[i_child]
            cov = gmm.covariances_[i_child]
            cond_prob = gmm.weights_[i_child]
            prob = parent.prob * cond_prob
            child_node_params = means, cov, cond_prob, prob
            self.add_node(child_node_params, parent)

    def get_gans(self, k_clusters):
        nodes = {self.nodes[0]}
        for i in range(k_clusters - 1):
            split_node = self.nodes[self.split_history[i]]
            nodes.remove(split_node)
            for child_node in split_node.child:
                nodes.add(child_node)
        return GANSet(self.session, list(nodes), self.nodes[0])

    def _recursive_shutdown(self, node_id):
        for child_node_id in self.nodes[node_id].child:
            self._recursive_shutdown(child_node_id)
        model = self.nodes[node_id].model
        model.session.close()

    def shutdown(self):
        self._recursive_shutdown(0)
