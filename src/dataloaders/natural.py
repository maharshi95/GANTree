import torch as tr

from base.dataloader import BaseDataLoader
from torchvision.datasets import CIFAR10, STL10


def normalize_mnist_images(x):
    return 2 * (x.type(tr.float32) / 255.0) - 1.0


class CIFARDataLoader(BaseDataLoader):

    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True):
        super(CIFARDataLoader, self).__init__((32, 32), None, train_batch_size, test_batch_size, get_tensor,
                                              supervised=True)

    def get_data(self):
        CIFAR10('../data/cifar10', download=True)

        train_data, train_labels = tr.load('../data/cifar10/processed/training.pt')
        test_data, test_labels = tr.load('../data/cifar10/processed/test.pt')

        train_data = normalize_mnist_images(train_data)
        test_data = normalize_mnist_images(test_data)
        return train_data, test_data, train_labels, test_labels


class STLDataLoader(BaseDataLoader):

    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True):
        super(STLDataLoader, self).__init__((32, 32), None, train_batch_size, test_batch_size, get_tensor,
                                            supervised=True)

    def get_data(self):
        STL10('../data/stl10', download=True)
        train_data, train_labels = tr.load('../data/stl10/processed/training.pt')
        test_data, test_labels = tr.load('../data/stl10/processed/test.pt')

        train_data = normalize_mnist_images(train_data)
        test_data = normalize_mnist_images(test_data)
        return train_data, test_data, train_labels, test_labels
