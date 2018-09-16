from __future__ import division
import tensorflow as tf


def scaled_tanh(x, scale=1.0):
    return scale * tf.tanh(x / scale)


def get_scaled_tanh(scale):
    return lambda x: scaled_tanh(x, scale)


