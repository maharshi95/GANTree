import numpy as np

from dataloaders.base import BaseDataLoader
from utils import np_utils
from dataset_loader import Dataset

class MNISTDataLoader(object):
    def __init__(self):
        self.dataset = Dataset('mnist', shuffle=True)
        self.num_samples = 128

    def mnist(self,  train_ratio=0.8):
        new_data = self.dataset.next_batch(batch_size = self.num_samples)
        n_train = int(train_ratio * new_data.shape[0])
        training, test = new_data[:n_train, :], new_data[n_train:, :]

        return training, test

    def get_z_dist(self, n_samples, dim=100, dist_type='normal'):
        if dist_type == 'uniform':
            return np.random.uniform(-1, 1, (n_samples, dim))
        if dist_type == 'normal':
            return np.random.normal(0, 1, (n_samples, dim))
        if dist_type == 'sphere':
            return np_utils.unit_norm(np.random.normal(0, 1, (n_samples, dim)), axis=-1)
        raise Exception('Invalid dist_type: {}'.format(dist_type))
