from __future__ import division
import tensorflow as tf
from tensorflow.contrib import layers


def scaled_tanh(x, scale=1.0):
    return scale * tf.tanh(x / scale)


def get_scaled_tanh(scale):
    return lambda x: scaled_tanh(x, scale)


def dense(inputs, num_outputs, activation_fn=tf.nn.relu):
    return layers.fully_connected(inputs, num_outputs, activation_fn)


def n_layers_dense(inputs, n_units, activations=None, name='n_layers_fully_connected'):
    n_layers = len(n_units)
    assert len(n_units) == len(activations)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        next_layer = inputs
        for i in range(n_layers):
            scope = 'layer_%d' % i
            act_fn = activations[i]
            next_layer = layers.fully_connected(next_layer, n_units[i], activation_fn=act_fn, scope=scope)
    return next_layer