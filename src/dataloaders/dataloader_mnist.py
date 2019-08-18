import numpy as np

from utils import np_utils
from exp_context import ExperimentContext

H = ExperimentContext.get_hyperparams()


class MNISTDataLoader(object):
    # def __init__(self):
    # self.dataset = Dataset('mnist', shuffle=True)

    def _scale(self, x, feature_range=(-1, 1)):
        """
        Scale the images to have pixel values between -1 and 1
        """
        # scale to feature_range
        min, max = feature_range
        x = x * (max - min) + min
        return x

    def next_batch(self, training=True):
        """
        Return the next batch for the training loop or testing
        """
        if training:
            # print("Dataset shuffled successfully.")
            training_data = np.load('./data/train.npy')
            len_train_set = len(training_data)
            idx = np.arange(len_train_set)
            np.random.shuffle(idx)
            training_data = training_data[idx]
            ii = np.random.randint(0, len_train_set - H.batch_size_train, size=1)[-1]
            x = training_data[ii:ii + H.batch_size_train]

        else:
            testing_data = np.load('./data/test.npy')
            len_test_set = len(testing_data)
            ii = np.random.randint(0, len_test_set - H.batch_size_test, size=1)[-1]
            x = testing_data[ii:ii + H.batch_size_test]

        return self._scale(x)

    def get_data(self):
        training = self.next_batch(training=True)
        test = self.next_batch(training=False)
        return training, test

    def get_z_dist(self, n_samples, dim=100, dist_type='normal'):
        if dist_type == 'uniform':
            return np.random.uniform(-1, 1, (n_samples, dim))
        if dist_type == 'normal':
            return np.random.normal(0, 1, (n_samples, dim))
        if dist_type == 'sphere':
            return np_utils.unit_norm(np.random.normal(0, 1, (n_samples, dim)), axis=-1)
        raise Exception('Invalid dist_type: {}'.format(dist_type))
