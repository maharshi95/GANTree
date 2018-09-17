import tensorflow as tf
from tensorflow.contrib import layers
from exp_context import ExperimentContext
from . import commons

H = ExperimentContext.Hyperparams


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


def encoder(x):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        n_units = [128, 128, 64, 64, H.z_size]
        n_layers = len(n_units)
        activations = [tf.nn.elu] * (n_layers - 1) + [commons.get_scaled_tanh(4.0)]
        z = n_layers_dense(x, n_units, activations)
        return z


def decoder(z):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        n_units = [64, 64, 128, 128, H.input_size]
        n_layers = len(n_units)
        activations = [tf.nn.elu] * (n_layers - 1) + [None]
        x = n_layers_dense(z, n_units, activations)
        return x


def disc(x):
    with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
        n_units = [64, 64, 128, 128, 1]
        n_layers = len(n_units)
        act_fn = [tf.nn.elu] * (n_layers - 1) + [None]
        logits = n_layers_dense(x, n_units, act_fn)
        return logits


def disc_v2(x):
    with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
        n_units = [64, 64, 128, 128, 1]
        n_layers = len(n_units)
        act_fn = [tf.nn.elu] * (n_layers - 1) + [None]
        logits = n_layers_dense(x, n_units, act_fn)
        logits_reshaped = tf.reshape(logits, [-1, H.logit_batch_size])
        entropy_logits = layers.fully_connected(logits_reshaped, num_outputs=1, activation_fn=None, scope='entropy_logits')
        return logits, entropy_logits