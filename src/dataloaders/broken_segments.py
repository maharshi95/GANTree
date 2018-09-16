import numpy as np

from dataloaders.base import BaseDataLoader


class BrokenSegmentsDataLoader(BaseDataLoader):
    def __init__(self, input_size=1, latent_size=1):
        super(BrokenSegmentsDataLoader, self).__init__(input_size, latent_size)

    def get_data(self, radius=1.0, train_ratio=0.8):
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
