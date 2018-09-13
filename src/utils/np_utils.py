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
