import numpy as np
from utils import np_utils
from utils.decorators import tensor_output

from configs import Config


class BaseDataLoader(object):

    def __init__(self, input_size=2, latent_size=2, train_batch_size=32, test_batch_size=32, get_tensor=True, supervised=False):
        self.input_size = input_size
        self.latent_size = latent_size
        self.get_tensor = True
        self.supervised = supervised

        if supervised:
            train_data, test_data, train_labels, test_labels = self.get_data()
            self.labels = {
                'train': train_labels,
                'test': test_labels,
            }

        else:
            train_data, test_data = self.get_data()

        self.batch_size = {
            'train': train_batch_size,
            'test': test_batch_size
        }

        self.data = {
            'train': train_data,
            'test': test_data
        }

        self.n_batches = {
            split: self.data[split].shape[0] // self.batch_size[split]
            for split in ['train', 'test']
        }

        self.batch_index = {
            'train': 0,
            'test': 0
        }

        self.shuffle('train')
        self.shuffle('test')

    def shuffle(self, split):
        n = len(self.data[split])
        perm = np.random.permutation(n)
        self.data[split] = self.data[split][perm]
        if self.supervised:
            self.labels[split] = self.labels[split][perm]

    @tensor_output(use_gpu=Config.use_gpu)
    def next_batch(self, split):
        start = self.batch_index[split] * self.batch_size[split]
        end = start + self.batch_size[split]
        self.batch_index[split] = (self.batch_size[split] + 1) % self.n_batches[split]

        if split == 'train' and self.batch_index[split] == 0:
            self.shuffle(split)

        data = self.data[split][start: end]

        if self.supervised:
            labels = self.labels[split][start: end]
            return data, labels
        else:
            return data

    @tensor_output(use_gpu=Config.use_gpu)
    def random_batch(self, split, batch_size):
        data = self.data[split]
        indices = np.random.permutation(len(data))[:batch_size]
        batch = data[indices]

        if self.supervised:
            labels = self.labels[split][indices]
            return batch, labels
        else:
            return batch

    def get_data(self):
        return NotImplemented

    @tensor_output(use_gpu=Config.use_gpu)
    def get_full_space(self, n_samples=1000, bounds=4.0):
        return np.random.uniform(-bounds, bounds, (n_samples, self.input_size))

    @tensor_output(use_gpu=Config.use_gpu)
    def get_z_dist(self, n_samples, dist_type, bounds=1):
        if dist_type == 'uniform':
            data = np.random.uniform(-bounds, bounds, (n_samples, self.latent_size))
        elif dist_type == 'normal':
            data = np.random.normal(0, 1, (n_samples, self.latent_size))
        elif dist_type == 'sphere':
            data = np_utils.unit_norm(np.random.normal(0, 1, (n_samples, self.latent_size)), axis=-1)
        else:
            raise Exception('Invalid dist_type: {}'.format(dist_type))
        return data

    @tensor_output(use_gpu=Config.use_gpu)
    def complete_data(self):
        return np.concatenate((self.data['train'], self.data['test']), axis=0)

    @tensor_output(use_gpu=Config.use_gpu)
    def train_data(self):
        return self.data['train']
