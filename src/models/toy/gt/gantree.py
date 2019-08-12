import logging
from collections import deque

import numpy as np
from torch import nn

from base.model import BaseGan
from base.model import BaseModel

from .gnode import GNode
from .utils import GNodeUtils
from .named_tuples import DistParams

logger = logging.getLogger(__name__)


class GanTree(BaseModel):
    GanModel = None  # type: BaseGan

    @classmethod
    def create_from_root(cls, root, name, hyperparams):
        # type: (GNode, str, object) -> GanTree
        nodes = {}
        q = deque([root])
        n_child = 0
        while q:
            node = q.popleft()
            nodes[node.id] = node
            n_child = max(n_child, len(node.child_ids))
            for cid in node.child_ids:
                q.append(node.child_nodes[cid])

        tree = GanTree(name, root.model_class, hyperparams, n_child=n_child)

        for cid in sorted(nodes.keys()):
            tree.nodes.append(nodes[cid])

        for i, node in enumerate(tree.nodes):
            assert node.id == i
        return tree

    def __init__(self, name, GanModel, hyperparams, x_batch=None, n_child=2):
        super(GanTree, self).__init__()
        self.name = name
        self.GanModel = GanModel
        self.H = hyperparams
        self.x_batch = x_batch
        self.n_child = n_child

        self.root_id = 0
        self.nodes = nn.ModuleList()  # type: list[GNode]
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

    @property
    def id_graph(self):
        def get_graph(node):
            if node.is_leaf:
                return None
            return {
                cnode.id: get_graph(cnode)
                for cnode in node.child_nodes
            }

        return {self.root.id: get_graph(self.root)}

    def __getitem__(self, item):
        return self.nodes[item]

    def __iter__(self):
        return iter(self.nodes)

    def create_child_node(self, dist_params, model, parent=None):
        # type: (DistParams, BaseGan, GNode | None) -> GNode
        new_node_id = len(self.nodes)
        new_node = GNodeUtils.create_child_node(new_node_id, dist_params, model, parent)
        self.nodes.append(new_node)
        return new_node

    def split_node(self, parent, x_batch=None, fixed=True, applyPCA = True, H = None):
        # type: (GNode, np.ndarray, bool) -> list[GNode]
        x_batch = self.x_batch if x_batch is None else x_batch

        child_nodes = GNodeUtils.split_node(parent, self.n_child, x_batch, base_id=len(self.nodes), applyPCA = applyPCA, H = H)

        self.nodes.extend(child_nodes)
        self.split_history.append(parent.id)
        parent.set_optimizer()

        return child_nodes

    def load_children(self, parent, path):
        child_nodes = GNodeUtils.load_children(parent, self.n_child, path, base_id = len(self.nodes))

        self.nodes.extend(child_nodes)
        self.split_history.append(parent.id)
        parent.set_optimizer()

        return child_nodes