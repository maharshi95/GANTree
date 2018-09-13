import numpy as np
import tensorflow as tf
from sklearn import mixture
from exp_context import ExperimentContext


class GNode(object):
    def __init__(self, node_id=-1, model=None, parent_id=-1):
        self.model = model
        self.cond_prob = 0.
        self.prob = 0.
        self.means = None
        self.cov = None
        self.child = []
        self.node_id = node_id
        self.parent_id = parent_id

    def __repr__(self):
        return '<GNode[name={} node_id={} parent_id={}]>'.format(self.name, self.node_id, self.parent_id)

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


class GANSet(object):
    def __init__(self, session, gan_nodes):
        self.gans = gan_nodes
        self.session = session

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
        self.add_node(params, parent_id=-1)
        self._is_initiated = True

    def parent(self, gnode):
        return self.nodes[gnode.parent_id]

    @property
    def max_generators(self):
        return len(self.split_history) + 1

    def add_node(self, params, parent_id=-1):
        new_node_id = len(self.nodes)
        model_name = "%s-%d" % (self.name, new_node_id)
        model = self.Model(model_name, session=self.session)
        model.build()
        model.initiate_service()

        if parent_id != -1:
            parent = self.nodes[parent_id]
            model.load_params_from_model(parent.model)
            parent.child.append(new_node_id)

        new_node = GNode(new_node_id, model, parent_id)
        new_node.params = params
        self.nodes.append(new_node)

    def split(self, parent_id):
        assert parent_id < len(self.nodes)
        assert parent_id not in self.split_history
        parent_node = self.nodes[parent_id]
        gmm = mixture.GaussianMixture(n_components=self.n_child, covariance_type='full', max_iter=1000)
        model = parent_node.model

        z_batch = model.encode(self.x_batch)
        gmm.fit(z_batch)
        self.mixture_models[parent_id] = gmm
        self.split_history.append(parent_id)
        for i_child in range(self.n_child):
            means = gmm.means_[i_child]
            cov = gmm.covariances_[i_child]
            cond_prob = gmm.weights_[i_child]
            prob = parent_node.prob * cond_prob
            child_node_params = means, cov, cond_prob, prob
            self.add_node(child_node_params, parent_id=parent_id)

    def get_gans(self, k_clusters):
        nodes = {self.nodes[0]}
        for i in range(k_clusters - 1):
            split_node_id = self.split_history[i]
            split_node = self.nodes[split_node_id]
            nodes.remove(split_node)
            for child_node_id in split_node.child:
                nodes.add(self.nodes[child_node_id])
        return GANSet(self.session, list(nodes))

    def _recursive_shutdown(self, node_id):
        for child_node_id in self.nodes[node_id].child:
            self._recursive_shutdown(child_node_id)
        model = self.nodes[node_id].model
        model.session.close()

    def shutdown(self):
        self._recursive_shutdown(0)
