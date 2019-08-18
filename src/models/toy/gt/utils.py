import logging
import torch as tr
import numpy as np
from termcolor import colored

from base.model import BaseGan
from .named_tuples import DistParams
from .gnode import GNode

logger = logging.getLogger(__name__)


# Named Tuple


class GNodeUtils:

    @staticmethod
    def split_models(model, n_child, name_func, fixed=True):
        # type: (BaseGan, int, callable, bool) -> list[BaseGan]
        """
        Creates ``n_child`` child models each having a shared encoder and dedicated decoder, disc pair.
        """

        encoder = model.encoder.copy()

        child_models = [
            model.__class__(name=name_func(i),
                            z_op_params=2,
                            encoder=encoder,
                            decoder=model.decoder.copy(),
                            disc_x=model.disc_x.copy(),
                            disc_z=model.disc_z.copy(),
                            z_bounds=model.z_bounds)
            for i in range(n_child)
        ]
        return child_models

    @staticmethod
    def create_child_node(node_id, dist_params, model, parent=None):
        # type: (int, DistParams, BaseGan, GNode | None) -> GNode
        new_node = GNode(node_id, model, parent)
        new_node.dist_params = dist_params

        logger.info(colored('Child Node %s created from %s' % (model.name, parent and parent.name), 'green', attrs=['bold']))
        return new_node

    @staticmethod
    def split_node(parent, n_child, x_batch, base_id, applyPCA = True, H = None):
        # type: (GNode, int, np.ndarray, int, bool) -> list[GNode]

        n_dim = parent.prior_means.shape[-1]

        logger.info('Starting Split Process: %s' % parent)
        
        parent.init_child_params(x_batch, n_components = 2, applyPCA = applyPCA, H = H)
    
        child_nodes = []

        Model = parent.gan.__class__

        common_encoder = parent.gan.encoder.copy(n_dim, parent.gan.channel)

        for i_child in range(n_child):
            node_id = base_id + i_child
            model_name = "node%d" % node_id
            logger.info('child id:%d' % i_child)

            
            means = parent.kmeans.means[i_child]
            cov = parent.kmeans.covs[i_child]


            child_model = Model(name = model_name,
                                z_op_params = (tr.Tensor(means), tr.Tensor(cov)),
                                encoder = common_encoder,
                                generator = parent.gan.generator.copy(n_dim, parent.gan.channel),
                                discriminator = parent.gan.discriminator.copy(n_dim)
                                )
            

            cond_prob = parent.kmeans.weights[i_child]

            prob = parent.prob * cond_prob
            child_node_params = DistParams(means, cov, cond_prob, prob)
            child_node = GNodeUtils.create_child_node(node_id, child_node_params, child_model, parent)
            child_nodes.append(child_node)

        return child_nodes

    @staticmethod
    def load_children(parent, n_child, path, base_id):
        # type: (GNode, int, np.ndarray, int, bool) -> list[GNode]

        n_dim = parent.prior_means.shape[-1]

        logger.info('Starting Loading Process: %s' % parent)
            
        child_nodes = []

        Model = parent.gan.__class__

        common_encoder = parent.gan.encoder.copy(n_dim, parent.gan.channel)

        for i_child in range(n_child):
            node_id = base_id + i_child
            model_name = "node%d" % node_id
            logger.info('child id:%d' % i_child)
            
            means = parent.kmeans.means[i_child]
            cov = parent.kmeans.covs[i_child]


            child_model = Model(name = model_name,
                                z_op_params = (tr.Tensor(means), tr.Tensor(cov)),
                                encoder = common_encoder,
                                generator = parent.gan.generator.copy(n_dim,  parent.gan.channel),
                                discriminator = parent.gan.discriminator.copy(n_dim)
                                )
            

            cond_prob = parent.kmeans.weights[i_child]

            prob = parent.prob * cond_prob
            child_node_params = DistParams(means, cov, cond_prob, prob)
            child_node = GNodeUtils.create_child_node(node_id, child_node_params, child_model, parent)
            child_node = GNode.load(path + str(i_child) + '_mnist_phase_1_2.pickle', child_node)
            child_nodes.append(child_node)

        return child_nodes