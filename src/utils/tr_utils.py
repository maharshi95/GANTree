import torch as tr
import numpy as np

from configs import Config


def as_np(v, dtype=None):
    if isinstance(v, list):
        return [as_np(u) for u in v]

    if isinstance(v, tuple):
        return tuple([as_np(u) for u in v])

    if not isinstance(v, tr.Tensor):
        return v

    val = v.detach()

    if Config.use_gpu:
        return val.cpu().numpy()

    val = val.numpy()
    if dtype is not None:
        val = val.astype(dtype)

    return val


def mu_cov(M):
    mu = tr.mean(M, dim=0, keepdim=True)
    cov = tr.mm((M - mu).t(), (M - mu)) / (len(M) - 1)
    return mu, cov


def ellipse_params(cov):
    cov = tr.Tensor(cov)

    trace = tr.trace(cov)
    det = tr.det(cov)
    a = tr.sqrt((trace + tr.sqrt(trace ** 2 - 4 * det)) / 2.0)
    b = tr.sqrt((trace - tr.sqrt(trace ** 2 - 4 * det)) / 2.0)
    theta = tr.atan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1]) / 2.0
    return theta, a, b


def rotate(X, theta):
    ct, st = tr.cos(theta), tr.sin(theta)
    R = tr.Tensor([[ct, -st],
                   [st, ct]])
    return tr.mm(X, R.t())
