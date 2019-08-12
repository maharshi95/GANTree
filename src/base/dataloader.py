import numpy as np
import torch as tr
from utils import np_utils
from utils.decorators import tensor_output

from configs import Config


class BaseDataLoader(object):

    def __init__(self, img_size=2, latent_size=2, train_batch_size=64, test_batch_size=64, get_tensor=True, supervised=False, classes = None):
        self.img_size = img_size
        self.latent_size = latent_size
        self.get_tensor = True
        self.supervised = supervised
        self.classes = classes

        self.batch_size = {
            'train': train_batch_size,
            'test': test_batch_size
        }

        all_data = self.get_data()

        self.update_data(*all_data)
        

    def update_data(self, train_data, test_data, train_labels=None, test_labels=None):

        #TODO: Modify to make this generic for non cuda devices
        if self.supervised:
            self.labels = {
                'train': tr.tensor(train_labels),
                'test': tr.tensor(test_labels),
            }

        self.data = {
            'train': tr.tensor(train_data),
            'test': tr.tensor(test_data),
        }

        self.n_batches = {
            split: self.data[split].shape[0] // self.batch_size[split]
            for split in ['train', 'test']
        }

        self.batch_index = {
            'train': 0,
            'test': 0
        }

        # self.shuffle('train')
        # self.shuffle('test')

    def shuffle(self, split):
        n = len(self.data[split])
        perm = tr.randperm(n)
        self.data[split] = self.data[split][perm]
        if self.supervised:
            self.labels[split] = self.labels[split][perm]

    @tensor_output(use_gpu=Config.use_gpu)
    def next_batch(self, split):
        start = self.batch_index[split] * self.batch_size[split]
        end = start + self.batch_size[split]
        self.batch_index[split] = (self.batch_index[split] + 1) % self.n_batches[split]

        if self.batch_index[split] == 0:
            self.shuffle(split)

        data = self.data[split][start: end]

        if self.supervised:
            labels = self.labels[split][start: end]
            return data, labels, [i for i in range(start, end)]
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
    def train_data(self):
        return self.data['train']

    @tensor_output(use_gpu=Config.use_gpu)
    def train_data_labels(self):
        return self.labels['train']

    @tensor_output(use_gpu=Config.use_gpu)
    def test_data(self):
        return self.data['test']

    @tensor_output(use_gpu=Config.use_gpu)
    def test_data_labels(self):
        return self.labels['test']
