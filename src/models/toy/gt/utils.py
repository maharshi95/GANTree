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
    def split_node(parent, n_child, x_batch, base_id, fixed=False, applyPCA = True):
        # type: (GNode, int, np.ndarray, int, bool) -> list[GNode]

        n_dim = parent.prior_means.shape[-1]

        logger.info('Starting Split Process: %s' % parent)
        if fixed:
            cluster1_points = np.random.normal(3., 1., (1000, n_dim))
            cluster2_points = np.random.normal(-3., 1., (1000, n_dim))
            all_embeddings = np.concatenate([cluster1_points, cluster2_points])
            parent.fit_gmm(x_batch, max_iter=100, Z=all_embeddings, warm_start=False)
        else:
            parent.init_child_params(x_batch, n_components = 2, applyPCA = applyPCA)
        child_nodes = []

        Model = parent.gan.__class__
        # logger.info('n_dim created')
        common_encoder = parent.gan.encoder.copy()
        # logger.info('common_encoder created')
        for i_child in range(n_child):
            node_id = base_id + i_child
            model_name = "node%d" % node_id
            logger.info('child id:%d' % i_child)

            if fixed:
                means = parent.gmm.means_[i_child]
                cov = parent.gmm.covariances_[i_child]
            else:
                means = parent.kmeans.means[i_child]
                cov = parent.kmeans.covs[i_child]

            # logger.info('alalal')
            child_model = Model(name=model_name,
                                z_op_params=(tr.Tensor(means), tr.Tensor(cov)),
                                encoder=common_encoder,
                                decoder=parent.gan.decoder.copy(),
                                disc_x=parent.gan.disc_x.copy(),
                                disc_z=None, )
            # TODO
            # disc_z=parent.gan.disc_z.copy(), )

            # logger.info('child created:%d'%i_child)
            # z_bounds=parent.gan.z_bounds)

            if fixed:
                cond_prob = parent.cluster_probs[i_child]
            else:
                cond_prob = parent.kmeans.weights[i_child]

            prob = parent.prob * cond_prob
            child_node_params = DistParams(means, cov, cond_prob, prob)
            child_node = GNodeUtils.create_child_node(node_id, child_node_params, child_model, parent)
            child_nodes.append(child_node)
        return child_nodes

    @staticmethod
    def load_children(parent, n_child, phase1_epochs, base_id):
        # type: (GNode, int, np.ndarray, int, bool) -> list[GNode]

        n_dim = parent.prior_means.shape[-1]

        logger.info('Starting Split Process: %s' % parent)

        child_nodes = []

        Model = parent.gan.__class__
        # logger.info('n_dim created')
        common_encoder = parent.gan.encoder.copy()
        # logger.info('common_encoder created')
        for i_child in range(n_child):
            node_id = base_id + i_child
            model_name = "node%d" % node_id
            logger.info('child id:%d' % i_child)

            means = parent.kmeans.means[i_child]
            cov = parent.kmeans.covs[i_child]

            # logger.info('alalal')
            child_model = Model(name=model_name,
                                z_op_params=(tr.Tensor(means), tr.Tensor(cov)),
                                encoder=common_encoder,
                                decoder=parent.gan.decoder.copy(),
                                disc_x=parent.gan.disc_x.copy(),
                                disc_z=None, )
            # TODO
            # disc_z=parent.gan.disc_z.copy(), )

            # logger.info('child created:%d'%i_child)
            # z_bounds=parent.gan.z_bounds)

            cond_prob = parent.kmeans.weights[i_child]

            prob = parent.prob * cond_prob
            child_node_params = DistParams(means, cov, cond_prob, prob)
            child_node = GNodeUtils.create_child_node(node_id, child_node_params, child_model, parent)
            child_node = GNode.loadwithkmeans('best_child' + str(i_child) + '_phase1_mnistdc_'+ str(phase1_epochs) + '.pickle', child_node)
            child_nodes.append(child_node)
        return child_nodes