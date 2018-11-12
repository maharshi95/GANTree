from __future__ import absolute_import
import torch as tr

from base.dataloader import BaseDataLoader
from torchvision.datasets import MNIST, FashionMNIST


def normalize_mnist_images(x):
    x = x[:, None, :, :]
    return 2 * (x.type(tr.float32) / 255.0) - 1.0


class MnistDataLoader(BaseDataLoader):

    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True, supervised=True):
        super(MnistDataLoader, self).__init__((28, 28), None, train_batch_size, test_batch_size, get_tensor,
                                              supervised=True)

    def get_data(self):
        MNIST('../data/mnist', download=True)

        train_data, train_labels = tr.load('../data/mnist/processed/training.pt')
        test_data, test_labels = tr.load('../data/mnist/processed/test.pt')

        train_data = normalize_mnist_images(train_data)
        test_data = normalize_mnist_images(test_data)
        return train_data, test_data, train_labels, test_labels


class FashionMnistDataLoader(BaseDataLoader):

    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True):
        super(FashionMnistDataLoader, self).__init__((28, 28), None, train_batch_size, test_batch_size, get_tensor,
                                                     supervised=True)

    def get_data(self):
        FashionMNIST('../data/fashion', download=True)
        train_data, train_labels = tr.load('../data/fashion/processed/training.pt')
        test_data, test_labels = tr.load('../data/fashion/processed/test.pt')

        train_data = normalize_mnist_images(train_data)
        test_data = normalize_mnist_images(test_data)
        return train_data, test_data, train_labels, test_labels


class MixedMnistDataLoader(BaseDataLoader):
    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True):
        super(MixedMnistDataLoader, self).__init__((28, 28), None, train_batch_size, test_batch_size, get_tensor,
                                                   supervised=True)

    def get_data(self):
        FashionMNIST('../data/fashion', download=True)
        MNIST('../data/fashion', download=True)

        mnist_train_data, mnist_train_labels = tr.load('../data/mnist/processed/training.pt')
        mnist_test_data, mnist_test_labels = tr.load('../data/mnist/processed/test.pt')

        fmnist_train_data, fmnist_train_labels = tr.load('../data/fashion/processed/training.pt')
        fmnist_test_data, fmnist_test_labels = tr.load('../data/fashion/processed/test.pt')

        train_data = tr.cat([mnist_train_data, fmnist_train_data])
        train_labels = tr.cat([mnist_train_labels, 10 + fmnist_train_labels])
        test_data = tr.cat([mnist_test_data, fmnist_test_data])
        test_labels = tr.cat([mnist_test_labels, 10 + fmnist_test_labels])

        train_data = normalize_mnist_images(train_data)
        test_data = normalize_mnist_images(test_data)
        return train_data, test_data, train_labels, test_labels
