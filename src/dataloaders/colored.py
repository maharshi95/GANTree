from base.dataloader import BaseDataLoader
import torch as tr
import numpy as np


def normalize_images(x):
    return 2 * (x.astype('float32') / 255.0) - 1.0


class CelebDataloader(BaseDataLoader):

    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True, supervised=True):
        super(CelebDataloader, self).__init__((64, 64), None, train_batch_size, test_batch_size, get_tensor,
                                              supervised=True)

    def get_data(self):
        data = np.load('../data/faces.npy')
        images = np.array([a for a in data[:, 0]])
        labels = np.array([a for a in data[:, 1]])

        images = tr.tensor(images.transpose([0, 3, 1, 2]).astype('float32'))
        labels = tr.tensor(labels)
        images = normalize_images(images)

        train_data, train_labels = images, labels
        test_data, test_labels = images, labels

        return train_data, test_data, train_labels, test_labels


class BedDataLoader(BaseDataLoader):

    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True, supervised=True):
        super(BedDataLoader, self).__init__((64, 64), None, train_batch_size, test_batch_size, get_tensor,
                                            supervised=True)

    def get_data(self):
        images = np.load('../data/bedrooms.npy')
        labels = np.zeros(images.shape[0])

        images = tr.tensor(images.astype('float32'))
        labels = tr.tensor(labels)
        images = normalize_images(images)

        train_data, train_labels = images, labels
        test_data, test_labels = images, labels

        return train_data, test_data, train_labels, test_labels


class FaceBedDataLoader(BaseDataLoader):
    def __init__(self, train_batch_size=32, test_batch_size=32, get_tensor=True, supervised=True):
        super(FaceBedDataLoader, self).__init__((64, 64), None, train_batch_size, test_batch_size, get_tensor,
                                                supervised=True)

    def get_data(self):
        images = np.load('../data/bedrooms.npy')
        labels = 2 * np.ones(images.shape[0]).astype('float32')

        bed_labels = labels.astype('float32')
        bed_images = normalize_images(images)

        data = np.load('../data/faces.npy')
        images = np.array([a for a in data[:, 0]])
        labels = np.array([a for a in data[:, 1]])

        images = images.transpose([0, 3, 1, 2])[:, [2, 1, 0]]
        face_labels = labels.astype('float32')
        face_images = normalize_images(images)

        images = np.concatenate([face_images, bed_images])
        labels = np.concatenate([face_labels, bed_labels])

        perm = np.random.permutation(images.shape[0])

        train_data, train_labels = images[perm], labels[perm]
        test_data, test_labels = images[perm][:10000], labels[perm][:10000]

        return train_data, test_data, train_labels, test_labels
