from __future__ import absolute_import
import torch as tr

from base.dataloader import BaseDataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np

class MnistDataLoader(BaseDataLoader):

    def __init__(self, img_size = 2, train_batch_size=64, test_batch_size=64, get_tensor=True, supervised=True, classes = None):
        super(MnistDataLoader, self).__init__(img_size, None, train_batch_size, test_batch_size, get_tensor,
                                              supervised, classes)

    def get_data(self):
        train_dataset = MNIST('../data/mnist', train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(self.img_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
        test_dataset = MNIST('../data/mnist', train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(self.img_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))

        train_data = np.array([x[0].numpy() for x in train_dataset])
        train_labels = np.array([x[1].numpy() for x in train_dataset])

        test_data = np.array([x[0].numpy() for x in test_dataset])
        test_labels = np.array([x[1].numpy() for x in test_dataset])

        if self.classes:
            train_data = train_data[np.where(np.isin(train_labels, self.classes))]
            train_labels = train_labels[np.where(np.isin(train_labels, self.classes))]
            test_data = test_data[np.where(np.isin(test_labels, self.classes))]
            test_labels = test_labels[np.where(np.isin(test_labels, self.classes))]


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
    def __init__(self, img_size = 2, train_batch_size=64, test_batch_size=64, get_tensor=True, supervised=True, classes = None):
        super(MixedMnistDataLoader, self).__init__(img_size, None, train_batch_size, test_batch_size, get_tensor,
                                              supervised, classes)

    def get_data(self):
        mnist_train_dataset = MNIST('../data/mnist', train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(self.img_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))

        mnist_test_dataset = MNIST('../data/mnist', train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(self.img_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))

        mnist_train_data = np.array([x[0].numpy() for x in mnist_train_dataset])
        mnist_train_labels = np.array([x[1].numpy() for x in mnist_train_dataset])

        mnist_test_data = np.array([x[0].numpy() for x in mnist_test_dataset])
        mnist_test_labels = np.array([x[1].numpy() for x in mnist_test_dataset])

        fashion_train_dataset = FashionMNIST('../data/fashion', train = True, download=True, 
                                              transform=transforms.Compose([
                                               transforms.Resize(self.img_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
        fashion_test_dataset = FashionMNIST('../data/fashion', train = False, download=True, 
                                              transform=transforms.Compose([
                                               transforms.Resize(self.img_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))

        fashion_train_data = np.array([x[0].numpy() for x in fashion_train_dataset])
        fashion_train_labels = np.array([x[1].numpy() for x in fashion_train_dataset])

        fashion_test_data = np.array([x[0].numpy() for x in fashion_test_dataset])
        fashion_test_labels = np.array([x[1].numpy() for x in fashion_test_dataset])

        train_data = np.concatenate((mnist_train_data, fashion_train_data))
        train_labels = np.concatenate((mnist_train_labels, 10 + fashion_train_labels))
        test_data = np.concatenate((mnist_test_data, fashion_test_data))
        test_labels = np.concatenate((mnist_test_labels, 10 + fashion_test_labels))

        return train_data, test_data, train_labels, test_labels
