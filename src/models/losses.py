import torch as tr
from torch import distributions as dist
from torch.nn import functional as F
from torch.nn.modules import loss

from utils import tr_utils
import numpy as np

from collections import Counter


def log_prob_sum(log_prob_1, log_prob_2):
    return tr.tensor(log_prob_1) + F.softplus(tr.tensor(log_prob_2) - tr.tensor(log_prob_1))


def x_clf_loss_assigned(mu1, sig1, w1, mu2, sig2, w2, z, preds=None):
    f1 = dist.MultivariateNormal(mu1, sig1)
    f2 = dist.MultivariateNormal(mu2, sig2)

    # p_z_m1 = f1.log_prob(z) + tr.log(w1)
    # p_z_m2 = f2.log_prob(z) + tr.log(w2)

    p_z_m1 = f1.log_prob(z)
    p_z_m2 = f2.log_prob(z)

    c = Counter(preds)
    
    if preds is None:
        loss = -tr.max(p_z_m1, p_z_m2) 
    else:
        preds = tr.tensor(preds, dtype=tr.int64).cuda()
        loss = -tr.where(preds == 0, p_z_m1, p_z_m2)
    
    weights = tr.Tensor([
            1.0 / np.maximum(c[0], 1e-9),
            1.0 / np.maximum(c[1], 1e-9)
        ]).cuda()[preds]

    loss = tr.sum(loss * weights)

    return loss


def x_clf_loss_unassigned(mu1, sig1, w1, mu2, sig2, w2, z, preds=None):

    f1 = dist.MultivariateNormal(mu1, sig1)
    f2 = dist.MultivariateNormal(mu2, sig2)

    p_z_m1 = f1.log_prob(z) + tr.log(w1)
    p_z_m2 = f2.log_prob(z) + tr.log(w2)

    loss = -tr.max(p_z_m1, p_z_m2)

    preds = [0 for i in range(len(p_z_m1))]

    for i in (np.where(loss == -p_z_m2)[0]):
        preds[i] = 1

    c = Counter(preds)

    count1 = len(np.where(loss == -p_z_m1)[0])
    count2 = len(np.where(loss == -p_z_m2)[0])

    weights = tr.Tensor([
            1.0 / np.maximum(count1, 1e-9),
            1.0 / np.maximum(count2, 1e-9)
        ]).cuda()[preds]

    loss = tr.sum(loss * weights)

    return loss


def x_clf_cross_loss(mu1, sig1, w1, mu2, sig2, w2, z, preds=None):
    f1 = dist.MultivariateNormal(mu1, sig1)
    f2 = dist.MultivariateNormal(mu2, sig2)

    p_z_m1 = f1.log_prob(z) + tr.log(w1)
    p_z_m2 = f2.log_prob(z) + tr.log(w2)

    p_z = log_prob_sum(p_z_m1, p_z_m2)

    if preds is None:
        loss = -tr.max(p_z_m1, p_z_m2) + p_z
    else:
        preds = tr.tensor(preds, dtype=tr.int32).cuda()
        loss = -tr.where(preds == 0, p_z_m1, p_z_m2) + p_z

    loss = loss.sum()
    return loss
