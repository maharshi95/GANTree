import numpy as np
import torch as tr
from utils import np_utils


def tensor_output(get_tensor=True):
    def decorator(f):
        def inner(*args, **kwargs):
            ret = f(*args, **kwargs)
            return tr.Tensor(ret) if get_tensor else ret

        return inner

    return decorator


class BaseDataLoader(object):

    def __init__(self, input_size=2, latent_size=2, train_batch_size=32, test_batch_size=32, get_tensor=True):
        self.input_size = input_size
        self.latent_size = latent_size
        self.get_tensor = True

        train_data, test_data = self.get_data()

        np.random.shuffle(train_data)
        np.random.shuffle(test_data)

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

    @tensor_output(True)
    def next_batch(self, split):
        start = self.batch_index[split] * self.batch_size[split]
        end = start + self.batch_size[split]
        self.batch_index[split] = (self.batch_size[split] + 1) % self.n_batches[split]

        if split == 'train' and self.batch_index[split] == 0:
            np.random.shuffle(self.data[split])

        data = self.data[split][start: end]
        return data

    def get_data(self):
        return NotImplemented

    @tensor_output(True)
    def get_full_space(self):
        return np.random.uniform(-10, 10, (1000, self.input_size))

    @tensor_output(True)
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
