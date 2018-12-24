import torch as tr
from torch import distributions as dist
from torch.nn import functional as F
from torch.nn.modules import loss

from utils import tr_utils
import numpy as np

from collections import Counter


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

def hinge_loss(mu1, mu2, margin = 20):
    hinge_loss = max(0, margin - np.linalg.norm(mu1 - mu2))
    return hinge_loss


# def x_clf_loss_assigned(mu1, sig1, w1, mu2, sig2, w2, z, preds=None):
#     f1 = dist.MultivariateNormal(mu1, sig1)
#     f2 = dist.MultivariateNormal(mu2, sig2)

#     # p_z_m1 = f1.log_prob(z) + tr.log(w1)
#     # p_z_m2 = f2.log_prob(z) + tr.log(w2)

#     p_z_m1 = f1.log_prob(z)
#     p_z_m2 = f2.log_prob(z)


#     if preds is None:
#         loss = -tr.max(p_z_m1, p_z_m2) 
#     else:
#         preds = tr.tensor(preds, dtype=tr.int32)
#         loss = -tr.where(preds == 0, p_z_m1, p_z_m2)

#     loss = loss.mean()
#     return loss

def x_clf_loss_assigned(mu1, sig1, w1, mu2, sig2, w2, z, preds=None):
    f1 = dist.MultivariateNormal(mu1, sig1)
    f2 = dist.MultivariateNormal(mu2, sig2)

    # p_z_m1 = f1.log_prob(z) + tr.log(w1)
    # p_z_m2 = f2.log_prob(z) + tr.log(w2)

    p_z_m1 = f1.log_prob(z)
    p_z_m2 = f2.log_prob(z)

    c = Counter(preds)
    # print(c[0])
    # print(c[1])
    
    if preds is None:
        loss = -tr.max(p_z_m1, p_z_m2) 
    else:
        preds = tr.tensor(preds, dtype=tr.int64)
        loss = -tr.where(preds == 0, p_z_m1, p_z_m2)
    
    weights = tr.Tensor([
            1.0 / np.maximum(c[0], 1e-9),
            1.0 / np.maximum(c[1], 1e-9)
        ])[preds]

    loss = tr.sum(loss * weights)

    return loss

def x_clf_loss_assigned_separate(mu1, sig1, w1, mu2, sig2, w2, z, preds=None):
    f1 = dist.MultivariateNormal(mu1, sig1)
    f2 = dist.MultivariateNormal(mu2, sig2)

    p_z_m1 = f1.log_prob(z)
    p_z_m2 = f2.log_prob(z)

    loss_ch0 = []
    loss_ch1 = []

    for i in range(len(preds)):
        if preds[i] == 0:
            loss_ch0.append(p_z_m1.detach().cpu().numpy()[0])
        elif preds[i] == 1:
            loss_ch1.append(p_z_m2.detach().cpu().numpy()[0])

    loss_ch0_ = -np.mean(np.asarray(loss_ch0))
    loss_ch1_ = -np.mean(np.asarray(loss_ch1))

    return loss_ch0_, loss_ch1_

def x_clf_loss_unassigned(mu1, sig1, w1, mu2, sig2, w2, z, preds=None):

    f1 = dist.MultivariateNormal(mu1, sig1)
    f2 = dist.MultivariateNormal(mu2, sig2)

    p_z_m1 = f1.log_prob(z) + tr.log(w1)
    p_z_m2 = f2.log_prob(z) + tr.log(w2)

    # p_z_m1 = f1.log_prob(z)
    # p_z_m2 = f2.log_prob(z)

    # c = Counter(preds)

    # p_z = log_prob_sum(p_z_m1, p_z_m2)

    # loss_1 = -tr.max(p_z_m1 - p_z, p_z_m2 - p_z) 
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
        ])[preds]

    loss = tr.sum(loss * weights)

    # loss = loss.mean()

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
        preds = tr.tensor(preds, dtype=tr.int32)
        loss = -tr.where(preds == 0, p_z_m1, p_z_m2) + p_z

    loss = loss.sum()
    return loss

def x_ce_k_child_loss(mu, sig, w, z, preds=None):
    k = mu.shape[0]
    p_z_m = []
    for i in range(k):
        f = dist.MultivariateNormal(mu[i], sig[i])
        p_z_mi = f.log_prob(z) + tr.log(w[i])
        p_z_m.append(p_z_mi)

    p_z = reduce(log_prob_sum, p_z_m)

    if preds is None:
        loss = -tr.max(p_z_m) + p_z
    else:
        preds = tr.tensor(preds, dtype=tr.int32)
        loss = -tr.where(preds == 0, p_z_m1, p_z_m2) + p_z

    loss = loss.mean()
    return loss


def kl_loss_parametric(mu1, mu2, sig1, sig2):
    m2 = mu2
    m1 = mu1

    loss = (tr.trace(sig2.inverse().mm(sig1)) + (m2 - m1).t().mm(sig2.inverse()).mm(m2 - m1) - m1.shape[0] + tr.log(
        tr.det(sig2)) - tr.log(tr.det(sig1))) / 2.

    return loss


def x_clf_loss_fixed(mu1, sig1, w1, mu2, sig2, w2, z1, z2):
    f1 = dist.MultivariateNormal(mu1, sig1)
    f2 = dist.MultivariateNormal(mu2, sig2)

    # p_z1_m1 = f1.log_prob(z1) + tr.log(w1)
    # p_z1_m2 = f2.log_prob(z1) + tr.log(w2)
    #
    # p_z2_m1 = f1.log_prob(z2) + tr.log(w1)
    # p_z2_m2 = f2.log_prob(z2) + tr.log(w2)
    #
    # p_z1 = log_prob_sum(p_z1_m1, p_z1_m2)
    # p_z2 = log_prob_sum(p_z2_m1, p_z2_m2)
    #
    # p_m1_z1 = p_z1_m1 - p_z1
    # p_m2_z2 = p_z2_m2 - p_z2
    #
    # loss = - p_m1_z1 - p_m2_z2

    # loss = - f1.log_prob(z1) - f2.log_prob(z2)

    # loss = loss.mean()

    z_mu1, z_sig1 = tr_utils.mu_cov(z1)
    z_mu2, z_sig2 = tr_utils.mu_cov(z2)

    loss = kl_loss_parametric(mu1[:, None], z_mu1.t(), sig1, z_sig1) + kl_loss_parametric(mu2[:, None], z_mu2.t(), sig2, z_sig2)

    return loss


def ls_gan_loss(logits, labels):
    return tr.mean((logits - labels) ** 2)


def sigmoid_cross_entropy_loss(logits, labels):
    if labels == 0.:
        labels = tr.zeros_like(logits)
    elif labels == 1.:
        labels = tr.ones_like(logits)

    return F.binary_cross_entropy_with_logits(logits, labels)
    # losses = tr.max(logits, tr.zeros_like(logits)) - logits * labels + tr.log(1 + tr.exp(-tr.abs(logits)))
    # return losses.mean()


def l2_reg(params, l=0.0002):
    loss = 0.
    for param in params:
        loss += l * tr.sum(param ** 2)
    return loss
