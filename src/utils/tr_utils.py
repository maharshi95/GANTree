import torch as tr


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


