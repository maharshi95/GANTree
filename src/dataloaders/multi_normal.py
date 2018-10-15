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
    count_samples = 0
    for i in range(4):
        if i < 3:
            n_samples = int(ratio[i] * num_samples)
            count_samples += n_samples
        else:
            n_samples = num_samples - count_samples
        samples = np.random.multivariate_normal(np.array(means[i]), cov=np.eye(means.shape[-1]) * cov[i], size=n_samples)
        X.append(samples)

    X = np.concatenate(X)
    np.random.shuffle(X)
    return X


class FourGaussiansDataLoader(BaseDataLoader):
    def __init__(self, input_size=1, latent_size=1):
        super(FourGaussiansDataLoader, self).__init__(input_size, latent_size)

    def get_data(self, train_ratio=0.6):
        num_samples = 20000

        mu = np.array([[2, 2], [-1, -1], [-2, 2], [1, -1]])
        sigma = np.array([0.19, 0.09, 0.02, 0.07])
        ratio = np_utils.prob_dist([7, 3, 2, 5])

        X = generate_multi_gaussian_samples(means=mu, cov=sigma, ratio=ratio, num_samples=num_samples)

        n_train = int(train_ratio * X.shape[0])

        training, test = X[:n_train, :], X[n_train:, :]

        return training, test


class FourSymGaussiansDataLoader(BaseDataLoader):
    def __init__(self, input_size=1, latent_size=1, train_batch_size=32, test_batch_size=32):
        super(FourSymGaussiansDataLoader, self).__init__(input_size, latent_size, train_batch_size, test_batch_size)

    def get_data(self, train_ratio=0.6):
        num_samples = 20000

        mu = np.array([[2, 2], [-2, -2], [-2, 2], [2, -2]])
        sigma = np.array([0.1, 0.1, 0.1, 0.1])
        ratio = np_utils.prob_dist([1, 1, 1, 1])

        X = generate_multi_gaussian_samples(means=mu, cov=sigma, ratio=ratio, num_samples=num_samples)

        n_train = int(train_ratio * X.shape[0])

        training, test = X[:n_train, :], X[n_train:, :]

        return training, test
