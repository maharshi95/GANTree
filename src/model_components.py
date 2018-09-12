import tensorflow as tf
from tensorflow.contrib import layers


def n_layers_fc(inputs, n_units, activations, name):
    n_layers = len(n_units)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        next_layer = inputs
        for i in range(n_layers):
            scope = 'layer_%d' % i
            act_fn = activations or activations[i]
            next_layer = layers.fully_connected(next_layer, n_units[i], activation_fn=act_fn, scope=scope)
    return next_layer


def encoder(x):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        fc1 = layers.fully_connected(x, 128, scope='fc1', reuse=tf.AUTO_REUSE)
        fc2 = layers.fully_connected(fc1, 128, scope='fc2', reuse=tf.AUTO_REUSE)
        fc3 = layers.fully_connected(fc2, 64, scope='fc3', reuse=tf.AUTO_REUSE)
        fc4 = layers.fully_connected(fc3, 1, activation_fn=tf.tanh, scope='fc4', reuse=tf.AUTO_REUSE)
        # [B, xsize]
        return fc4


def decoder(x):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        fc1 = layers.fully_connected(x, 64, scope='fc1', reuse=tf.AUTO_REUSE)
        fc2 = layers.fully_connected(fc1, 128, scope='fc2', reuse=tf.AUTO_REUSE)
        fc3 = layers.fully_connected(fc2, 2, activation_fn=tf.tanh, scope='fc3', reuse=tf.AUTO_REUSE)
        # [B, zsize]
        return fc3


def disc(x):
    with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
        fc1 = layers.fully_connected(x, 128, scope='fc1', reuse=tf.AUTO_REUSE)
        fc2 = layers.fully_connected(fc1, 128, scope='fc2', reuse=tf.AUTO_REUSE)
        fc3 = layers.fully_connected(fc2, 128, scope='fc3', reuse=tf.AUTO_REUSE)
        fc4 = layers.fully_connected(fc3, 128, scope='fc4', reuse=tf.AUTO_REUSE)
        # [B, 2]
        logits = layers.fully_connected(fc4, 2, activation_fn=None, scope='logits', reuse=tf.AUTO_REUSE)

        return logits
