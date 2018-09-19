from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def flatten(inputs):
    # type: (tf.Tensor) -> tf.Tensor
    """
    Reshapes a Tensor into a 2D tensor, retaining the batch (first) dimension.
    :param inputs: input Tensor
    :return: flattened Tensor
    """
    batch_size = tf.shape(inputs)[0]
    n_features = np.product(map(lambda x: x.value, inputs.shape[1:]))
    return tf.reshape(inputs, (batch_size, n_features))


def scaled_tanh(x, scale=1.0):
    return scale * tf.tanh(x / scale)


def get_scaled_tanh(scale):
    """
    :param scale: scale of tanh activation
    :return: an activation function which computes: scale * tanh(x / scale)
    """
    return lambda x: scaled_tanh(x, scale)


def dense(inputs, num_outputs, activation_fn=tf.nn.relu):
    # type: (tf.Tensor, int, object) -> tf.Tensor

    return layers.fully_connected(inputs, num_outputs, activation_fn)


def conv2d(inputs, num_filters, padding="SAME", kernel_size=5, stride=2, scope='conv2D'):
    layers.conv2d(inputs, num_filters, kernel_size, stride, padding, activation_fn=None, scope=scope)


def batch_norm(inputs, training, epsilon=1e-5, momentum=0.9):
    return tf.layers.batch_normalization(inputs, training=training, epsilon=epsilon, momentum=momentum)


def dropout(inputs, keep_prob=0.5, noise_shape=None, training=True):
    layers.dropout(inputs, keep_prob, noise_shape, is_training=training)


def n_layers_dense(inputs, n_units, activations=None, name='n_layers_dense'):
    n_layers = len(n_units)
    assert len(n_units) == len(activations)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        next_layer = inputs
        for i in range(n_layers):
            scope = 'layer_%d' % i
            act_fn = activations[i]
            next_layer = layers.fully_connected(next_layer, n_units[i], activation_fn=act_fn, scope=scope)
    return next_layer


def resnet_block(inputs, apply_block_fn):
    output = inputs + apply_block_fn(inputs)
    return output
