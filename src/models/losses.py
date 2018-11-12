import torch as tr
from torch import distributions as dist
from torch.nn import functional as F
from torch.nn.modules import loss


def x_clf_loss_v1(means1, cov1, means2, cov2, z1, z2):
    f1 = dist.MultivariateNormal(means1, cov1)
    f2 = dist.MultivariateNormal(means2, cov2)

    loss = (f2.log_prob(z1) - f1.log_prob(z1)).mean() + (f1.log_prob(z2) - f2.log_prob(z2)).mean()

    # f2_z1 = dist.exponential(f2.log_prob(z1))
    # f1_z1 = dist.exponential(f1.log_prob(z1))
    #
    # f2_z1_mask = tr.stack([p > 0.3 for p in f2_z1])
    # f1_z1_mask = tr.stack([p < 0.7 for p in f1_z1])

    # f2_z1_masked = sum(tr.mul(f2.log_prob(z1), f2_z1_mask))
    # f1_z1_masked = sum(tr.mul(f1.log_prob(z1), f1_z1_mask))
    #
    # f1_z2 = dist.exponential(f2.log_prob(z2))
    # f2_z2 = dist.exponential(f2.log_prob(z2))
    #
    # f1_z2_mask = tr.stack([p > 0.3 for p in f1_z2])
    # f2_z2_mask = tr.stack([p < 0.7 for p in f2_z2])

    # f1_z2_masked = sum(tr.mul(f1.log_prob(z2), f1_z2_mask))
    # f2_z2_masked = sum(tr.mul(f2.log_prob(z2), f2_z2_mask))

    # loss = (f2_z1_masked - f1_z1_masked).mean() + (f1_z2_masked - f2_z2_masked).mean()

    return loss


def log_prob_sum(log_prob_1, log_prob_2):
    return tr.tensor(log_prob_1) + F.softplus(tr.tensor(log_prob_2) - tr.tensor(log_prob_1))


def x_clf_loss(mu1, sig1, w1, mu2, sig2, w2, z):
    f1 = dist.MultivariateNormal(mu1, sig1)
    f2 = dist.MultivariateNormal(mu2, sig2)

    p_z_m1 = f1.log_prob(z) + tr.log(w1)
    p_z_m2 = f2.log_prob(z) + tr.log(w2)

    den = log_prob_sum(p_z_m1, p_z_m2)

    loss = -tr.max(p_z_m1, p_z_m2) + (den)

    loss = loss.mean()
    return loss


def x_clf_loss_fixed(mu1, sig1, w1, mu2, sig2, w2, z1, z2):
    f1 = dist.MultivariateNormal(mu1, sig1)
    f2 = dist.MultivariateNormal(mu2, sig2)

    p_z1_m1 = f1.log_prob(z1) + tr.log(w1)
    p_z1_m2 = f2.log_prob(z1) + tr.log(w2)

    p_z2_m1 = f1.log_prob(z2) + tr.log(w1)
    p_z2_m2 = f2.log_prob(z2) + tr.log(w2)

    p_z1 = p_z1_m1 + p_z1_m2
    p_z2 = p_z2_m1 + p_z2_m2

    p_m1_z1 = p_z1_m1 - p_z1
    p_m2_z2 = p_z2_m2 - p_z2

    loss = - p_m1_z1 - p_m2_z2

    loss = loss.mean()
    return loss


def sigmoid_cross_entropy_loss(logits, labels):
    if labels == 0.:
        labels = tr.zeros_like(logits)
    elif labels == 1.:
        labels = tr.ones_like(logits)

    losses = tr.max(logits, tr.zeros_like(logits)) - logits * labels + tr.log(1 + tr.exp(-tr.abs(logits)))
    return losses.mean()


def l2_reg(params, l=0.0002):
    loss = 0.
    for param in params:
        loss += l * tr.sum(param ** 2)
    return loss
