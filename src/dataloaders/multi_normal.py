import numpy as np
from base.dataloader import BaseDataLoader
from utils import np_utils


class TwoGaussiansDataLoader(BaseDataLoader):
    def __init__(self, input_size=1, latent_size=1):
        super(TwoGaussiansDataLoader, self).__init__(input_size, latent_size)

    def get_data(self, train_ratio=0.6):
        num_samples = 20000

        X1 = np.random.normal(0, 1.0, [num_samples // 2, self.input_size])
        X2 = np.random.normal(5, 0.6, [num_samples // 2, self.input_size])

        X = np.concatenate([X1, X2])
        np.random.shuffle(X)

        n_train = int(train_ratio * X.shape[0])

        training, test = X[:n_train, :], X[n_train:, :]

        return training, test


def generate_multi_gaussian_samples(means, cov, ratio, num_samples):
    X = []
    labels = []
    count_samples = 0
    n_modes = len(means)
    for i in range(n_modes):
        if i < n_modes - 1:
            n_samples = int(ratio[i] * num_samples)
            count_samples += n_samples
        else:
            n_samples = num_samples - count_samples
        samples = np.random.multivariate_normal(np.array(means[i]), cov=np.eye(means.shape[-1]) * cov[i], size=n_samples)
        X.append(samples)
        labels.extend([i] * n_samples)

    X = np.concatenate(X)
    labels = np.array(labels)
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    labels = labels[perm]
    return X, labels


class FourGaussiansDataLoader(BaseDataLoader):
    def __init__(self, input_size=1, latent_size=2, train_batch_size=32, test_batch_size=32, *args, **kwargs):
        super(FourGaussiansDataLoader, self).__init__(input_size, latent_size, train_batch_size, test_batch_size, *args,
                                                      **kwargs)

    def get_data(self, train_ratio=0.6):
        num_samples = 100000

        mu = np.array([[2, 2], [-1, -1], [-2, 2], [1, -1]])
        sigma = np.array([0.19, 0.09, 0.02, 0.07])
        ratio = np_utils.prob_dist([3, 3, 7, 5])

        X, labels = generate_multi_gaussian_samples(means=mu, cov=sigma, ratio=ratio, num_samples=num_samples)

        n_train = int(train_ratio * X.shape[0])

        training_data, test_data = X[:n_train, :], X[n_train:, :]
        training_labels, test_labels = labels[:n_train], labels[n_train:]

        return training_data, test_data, training_labels, test_labels


class FourSymGaussiansDataLoader(BaseDataLoader):
    def __init__(self, input_size=1, latent_size=2, train_batch_size=32, test_batch_size=32, *args, **kwargs):
        super(FourSymGaussiansDataLoader, self).__init__(input_size, latent_size, train_batch_size, test_batch_size, *args,
                                                         **kwargs)

    def get_data(self, train_ratio=0.6):
        num_samples = 20000

        np.random.seed(42)

        mu = np.array([[2, 2], [-2, -2], [-2, 2], [2, -2]])
        sigma = np.array([0.1, 0.1, 0.1, 0.1])
        ratio = np_utils.prob_dist([1, 1, 1, 1])

        X, labels = generate_multi_gaussian_samples(means=mu, cov=sigma, ratio=ratio, num_samples=num_samples)

        n_train = int(train_ratio * X.shape[0])

        training_data, test_data = X[:n_train, :], X[n_train:, :]
        training_labels, test_labels = labels[:n_train], labels[n_train:]

        return training_data, test_data, training_labels, test_labels


class NineSymGaussiansDataLoader(BaseDataLoader):
    def __init__(self, input_size=1, latent_size=2, train_batch_size=32, test_batch_size=32, *args, **kwargs):
        super(NineSymGaussiansDataLoader, self).__init__(input_size, latent_size, train_batch_size, test_batch_size, *args,
                                                         **kwargs)

    def get_data(self, train_ratio=0.6):
        num_samples = 100000

        means = [[i, j] for i in [-3, 0, 3] for j in [-3, 0, 3]]
        std = [0.15] * 9

        np.random.seed(42)

        mu = np.array(means)
        sigma = np.array(std)
        ratio = np_utils.prob_dist([1] * 9)

        X, labels = generate_multi_gaussian_samples(means=mu, cov=sigma, ratio=ratio, num_samples=num_samples)

        n_train = int(train_ratio * X.shape[0])

        training_data, test_data = X[:n_train, :], X[n_train:, :]
        training_labels, test_labels = labels[:n_train], labels[n_train:]

        return training_data, test_data, training_labels, test_labels


class NineGaussiansDataLoader(BaseDataLoader):
    def __init__(self, input_size=1, latent_size=2, train_batch_size=32, test_batch_size=32, *args, **kwargs):
        super(NineGaussiansDataLoader, self).__init__(input_size, latent_size, train_batch_size, test_batch_size, *args,
                                                      **kwargs)

    def get_data(self, train_ratio=0.6):
        num_samples = 100000

        means = [[i, j] for i in [-4, 0, 4] for j in [-4, 0, 4]]
        std = [0.15] * 9

        np.random.seed(42)

        mu = np.array(means) + np.random.uniform(-1, 1, (9, 2))
        sigma = np.array(std) * np.random.uniform(0.8, 1.2, 9)
        ratio = np_utils.prob_dist(np.random.uniform(3, 9, 9))

        X, labels = generate_multi_gaussian_samples(means=mu, cov=sigma, ratio=ratio, num_samples=num_samples)

        n_train = int(train_ratio * X.shape[0])

        training_data, test_data = X[:n_train, :], X[n_train:, :]
        training_labels, test_labels = labels[:n_train], labels[n_train:]

        return training_data, test_data, training_labels, test_labels
