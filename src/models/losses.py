import torch as tr
from torch import distributions as dist


def x_clf_loss(means1, cov1, means2, cov2, x1, x2):
    f1 = dist.MultivariateNormal(means1, cov1)
    f2 = dist.MultivariateNormal(means2, cov2)

    loss = (f2.log_prob(x1) - f1.log_prob(x1)).mean() + (f1.log_prob(x2) - f2.log_prob(x2)).mean()
    return loss


def sigmoid_cross_entropy_loss(logits, labels):
    if labels == 0.:
        labels = tr.zeros_like(logits)
    elif labels == 1.:
        labels = tr.ones_like(logits)

    losses = tr.max(logits, tr.zeros_like(logits)) - logits * labels + tr.log(1 + tr.exp(-tr.abs(logits)))
    return losses.mean()
