import logging

import numpy as np
from scipy.stats import norm as sp_norm
from scipy.stats import chi2

logger = logging.getLogger(__name__)


def prob_dist(A, axis=-1):
    A = np.array(A)
    if not np.all(A >= 0.):
        print(A.min())
    return A / np.maximum(A.sum(axis=axis, keepdims=True), + 1e-40)


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

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

# def ellipse_params(means, cov, scale=3):
    # trace = np.trace(cov)
    # det = np.linalg.det(cov)
    # a = np.sqrt((trace + np.sqrt(trace ** 2 - 4 * det)) / 2.0)
    # b = np.sqrt((trace - np.sqrt(trace ** 2 - 4 * det)) / 2.0)
    # theta = np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1]) / 2.0


def ellipse_params(means, cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * sp_norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    # print('xxxxxxxxxxxxxxxxxxxxxxxx')
    # print(val)
    # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    # vals, vecs = eigsorted(cov)
    # width, height = 2 * nsig * np.sqrt(vals)
    # rotation = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    return means, rotation, width, height
    # return means, theta, 2 * a * scale, 2 * b * scale


def rotate(X, theta):
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st],
                  [st, ct]])
    return np.matmul(X, R.T)
