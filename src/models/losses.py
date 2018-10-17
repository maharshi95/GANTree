import torch as tr
from torch import distributions as dist


def x_clf_loss(means1, cov1, means2, cov2, z1, z2):
    f1 = dist.MultivariateNormal(means1, cov1)
    f2 = dist.MultivariateNormal(means2, cov2)

    f2_z1 = tr.distributions.exponential(f2.log_prob(z1))
    f1_z1 = tr.distributions.exponential(f1.log_prob(z1))

    f2_z1_mask = tr.stack([p>0.3 for p in f2_z1])
    f1_z1_mask = tr.stack([p<0.7 for p in f1_z1])

    f2_z1_masked = sum(tr.mul(f2.log_prob(z1),f2_z1_mask))
    f1_z1_masked = sum(tr.mul(f1.log_prob(z1),f1_z1_mask))

    f1_z2 = tr.distributions.exponential(f2.log_prob(z2))
    f2_z2 = tr.distributions.exponential(f2.log_prob(z2))

    f1_z2_mask = tr.stack([p>0.3 for p in f1_z2])
    f2_z2_mask = tr.stack([p<0.7 for p in f2_z2])

    f1_z2_masked = sum(tr.mul(f1.log_prob(z2),f1_z2_mask))
    f2_z2_masked = sum(tr.mul(f2.log_prob(z2),f2_z2_mask))


    # loss = (f2.log_prob(z1) - f1.log_prob(z1)).mean() + (f1.log_prob(z2) - f2.log_prob(z2)).mean()

    loss = (f2_z1_masked-f1_z1_masked).mean() + (f1_z2_masked - f2_z2_masked).mean()

    print ('x_clf_loss',loss)
    print()
    return loss


def sigmoid_cross_entropy_loss(logits, labels):
    if labels == 0.:
        labels = tr.zeros_like(logits)
    elif labels == 1.:
        labels = tr.ones_like(logits)

    losses = tr.max(logits, tr.zeros_like(logits)) - logits * labels + tr.log(1 + tr.exp(-tr.abs(logits)))
    return losses.mean()
