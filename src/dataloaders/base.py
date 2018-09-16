import numpy as np
from utils import np_utils


class BaseDataLoader(object):

    def __init__(self, input_size=1, latent_size=1):
        self.input_size = input_size
        self.latent_size = latent_size

    def get_data(self):
        return NotImplemented

    def get_full_space(self):
        return np.random.uniform(-10, 10, (1000, self.input_size))

    def get_z_dist(self, n_samples, dist_type):
        if dist_type == 'uniform':
            return np.random.uniform(-1, 1, (n_samples, self.latent_size))
        if dist_type == 'normal':
            return np.random.normal(0, 1, (n_samples, self.latent_size))
        if dist_type == 'sphere':
            return np_utils.unit_norm(np.random.normal(0, 1, (n_samples, self.latent_size)), axis=-1)
        raise Exception('Invalid dist_type: {}'.format(dist_type))
