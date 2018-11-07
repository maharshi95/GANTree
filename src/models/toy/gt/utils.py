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
        print('')

        return new_node

    @staticmethod
    def split_node(parent, n_child, x_batch, base_id, fixed=True):
        # type: (GNode, int, np.ndarray, int, bool) -> list[GNode]

        logger.info('Starting Split Process: %s' % parent)
        parent.fit_gmm(x_batch, max_iter=1000)
        logger.info('Gaussian Mixture Fitted')
        print('')

        child_nodes = []

        # gan_models = GNodeUtils.split_models(parent.gan, n_child, fixed=fixed)

        Model = parent.gan.__class__
        n_dim = parent.prior_means.shape[-1]

        common_encoder = parent.gan.encoder.copy()

        for i_child in range(n_child):
            node_id = base_id + i_child
            model_name = "node%d" % node_id

            if fixed:
                means = np.ones(n_dim) * 3.0 * np.power(-1, i_child)
                cov = np.eye(n_dim)
            else:
                means = parent.gmm.means_[i_child]
                cov = parent.gmm.covariances_[-i_child]

            child_model = Model(name=model_name,
                                z_op_params=(tr.Tensor(means), tr.Tensor(cov)),
                                encoder=common_encoder,
                                decoder=parent.gan.decoder.copy(),
                                disc_x=parent.gan.disc_x.copy(),
                                disc_z=parent.gan.disc_z.copy(), )
            # z_bounds=parent.gan.z_bounds)

            cond_prob = parent.cluster_probs[i_child]
            prob = parent.prob * cond_prob
            child_node_params = DistParams(means, cov, cond_prob, prob)
            child_node = GNodeUtils.create_child_node(node_id, child_node_params, child_model, parent)
            child_nodes.append(child_node)
        return child_nodes
