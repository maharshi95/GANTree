import logging

import numpy as np
from scipy.stats import norm as sp_norm

logger = logging.getLogger(__name__)


def prob_dist(A, axis=-1):
    A = np.array(A, dtype=np.float32)
    assert all(A >= 0)
    return A / (A.sum(axis=axis, keepdims=True) + 1e-9)


def unit_norm(A, axis=-1):
    A = np.array(A)
    norm = np.linalg.norm(A, axis=axis, keepdims=True)
    return A / np.maximum(norm, 1e-15)


def one_hot(labels, n_classes):
    return np.eye(n_classes)[labels]


def shuffled_copy(inputs):
    return inputs[np.random.permutation(inputs.shape[0])]


def random_select(inputs, n_samples):
    return inputs[np.random.permutation(inputs.shape[0])[:n_samples]]


def ellipse_params(means, cov, scale=3):
    trace = np.trace(cov)
    det = np.linalg.det(cov)
    a = np.sqrt((trace + np.sqrt(trace ** 2 - 4 * det)) / 2.0)
    b = np.sqrt((trace - np.sqrt(trace ** 2 - 4 * det)) / 2.0)
    theta = np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1]) / 2.0
    return means, theta, 2 * a * scale, 2 * b * scale


def rotate(X, theta):
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st],
                  [st, ct]])
    return np.matmul(X, R.T)
