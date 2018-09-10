import tensorflow as tf
from tensorflow.contrib import layers


def encoder(x):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        fc1 = layers.fully_connected(x, 128, scope='fc1', reuse=tf.AUTO_REUSE)
        fc2 = layers.fully_connected(fc1, 64, scope='fc2', reuse=tf.AUTO_REUSE)
        fc3 = layers.fully_connected(fc2, 1, activation_fn=tf.tanh, scope='fc3', reuse=tf.AUTO_REUSE)
        # [B, xsize]
        return fc3


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