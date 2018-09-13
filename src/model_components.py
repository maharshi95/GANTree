import tensorflow as tf
from tensorflow.contrib import layers


def n_layers_fc(inputs, n_units, activations=None, name='n_layers_fully_connected'):
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
        n_units = [128, 128, 64, 64, 1]
        n_layers = len(n_units)
        activations = [tf.nn.relu] * (n_layers - 1) + [None]
        z = n_layers_fc(x, n_units, activations)
        return z


def decoder(z):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        n_units = [64, 64, 128, 128, 2]
        n_layers = len(n_units)
        activations = [tf.nn.leaky_relu] * (n_layers - 1) + [None]
        x = n_layers_fc(z, n_units, activations)
        return x


def disc(x):
    with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
        fc_outputs = n_layers_fc(x, [128] * 5, [tf.nn.relu] * 5)
        logits = layers.fully_connected(fc_outputs, 2, activation_fn=None, scope='logits', reuse=tf.AUTO_REUSE)
        return logits
