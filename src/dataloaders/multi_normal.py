import numpy as np

from dataloaders.base import BaseDataLoader


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
