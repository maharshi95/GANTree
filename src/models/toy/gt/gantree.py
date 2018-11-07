import logging
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

    def split_node(self, parent, x_batch=None, fixed=True):
        # type: (GNode, np.ndarray, bool) -> list[GNode]
        x_batch = self.x_batch if x_batch is None else x_batch

        child_nodes = GNodeUtils.split_node(parent, self.n_child, x_batch,
                                            base_id=len(self.nodes), fixed=fixed)

        self.nodes.extend(child_nodes)
        self.split_history.append(parent.id)
        parent.set_optimizer()

        return child_nodes
