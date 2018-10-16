from __future__ import division
import tensorflow as tf


def sigmoid_cross_entropy_loss(labels, logits, name=None):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name=name)
    )
