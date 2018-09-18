import numpy as np

from dataloaders.base import BaseDataLoader
from utils import np_utils
from dataset_loader import Dataset
from exp_context import ExperimentContext

H = ExperimentContext.get_hyperparams()


class MNISTDataLoader(object):
    def __init__(self, batch_size_train=32, batch_size_test=32, ):
        self.data = {
            'train': np.load('./data/train.npy'),
            'test': np.load('./data/test.npy')
        }
        self.batch_size = {
            'train': batch_size_train,
            'test': batch_size_test
        }
        self.current_batch = {
            'train': 0,
            'test': 0
        }

        self.n_batches = {
            'train': (len(self.data['train']) // self.batch_size['train']),
            'test': (len(self.data['test']) // self.batch_size['test'])
        }

    def shuffle(self, dataset):
        np.random.shuffle(dataset)

    def random_batch(self, split, batch_size):
        perm = np.random.permutation(len(self.data[split]))
        shuffled_data = self.data[split][perm]
        n_batches = len(shuffled_data) // batch_size
        i_batch = np.random.randint(0, n_batches)
        return shuffled_data[i_batch * batch_size:(i_batch + 1) * batch_size]

    def _scale(self, x, feature_range=(-1, 1)):
        """
        Scale the images to have pixel values between -1 and 1
        """
        # scale to feature_range
        min, max = feature_range
        x = x * (max - min) + min
        return x

    def next_batch(self, split):
        """
        Return the next batch for the training loop
        """

        batch_idx = self.current_batch[split]
        batch_size = self.batch_size[split]

        if batch_idx == 0:  # or split == 'train':
            self.shuffle(self.data[split])

        self.current_batch[split] = (self.current_batch[split] + 1) % self.n_batches[split]
        start = batch_idx * batch_size
        batch = self.data[split][start:start + batch_size]
        return self._scale(batch)

    def get_data(self):
        training = self.next_batch(split='train')
        test = self.next_batch(split='test')
        return training, test

    def get_z_dist(self, n_samples, dim=H.z_size, dist_type='normal'):
        if dist_type == 'uniform':
            return np.random.uniform(-1, 1, (n_samples, dim))
        if dist_type == 'normal':
            return np.random.normal(0, 1, (n_samples, dim))
        if dist_type == 'sphere':
            return np_utils.unit_norm(np.random.normal(0, 1, (n_samples, dim)), axis=-1)
        raise Exception('Invalid dist_type: {}'.format(dist_type))
