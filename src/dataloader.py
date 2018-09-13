import numpy as np
from utils import np_utils


class DataLoader(object):

    def broken_circle(self, radius=1.0, train_ratio=0.8):
        num_samples = 10000
        theta = np.random.uniform(0, 2 * np.pi, num_samples)

        filter_theta = [
            [0, 15],
            [345, 360],
            [75, 95],
            [120, 139],
            [200, 270]
        ]

        theta = filter(lambda x: any([r[0] <= x * 180 / np.pi <= r[1] for r in filter_theta]), theta)

        points = np.transpose(np.array([radius * np.cos(theta), radius * np.sin(theta)]))
        new_data = np.array(points)
        n_train = int(train_ratio * new_data.shape[0])
        training, test = new_data[:n_train, :], new_data[n_train:, :]

        return training, test

    def broken_segments(self, train_ratio=0.8):
        num_samples = 10000

        segments = [
            [[0, 1], [1, 0]],
            [[0.75, 0.75], [1.5, 1.5]],
            [[-0.25, 0.2], [0, -0.3]],
            [[-1.5, 0], [0, 1.5]],
            [[0.5, -1.5], [1.5, -0.5]]
        ]

        segments = np.array(segments)
        points = []
        for i in range(num_samples):
            s = segments[np.random.randint(0, segments.shape[0])]
            l = np.random.uniform(0, 1)
            p = s[0] * l + s[1] * (1 - l)
            points.append(p)
        data = np.array(points)
        n_train = int(train_ratio * data.shape[0])
        training, test = data[:n_train, :], data[n_train:, :]

        return training, test

    def get_z_dist(self, n_samples, dim=1, dist_type='normal'):
        if dist_type == 'uniform':
            return np.random.uniform(-1, 1, (n_samples, dim))
        if dist_type == 'normal':
            return np.random.normal(0, 1, (n_samples, dim))
        if dist_type == 'sphere':
            return np_utils.unit_norm(np.random.normal(0, 1, (n_samples, dim)), axis=-1)
        raise Exception('Invalid dist_type: {}'.format(dist_type))
