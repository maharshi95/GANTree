import tensorflow as tf
from tensorflow.contrib import layers
from exp_context import ExperimentContext
from model_components import commons
import numpy as np

H = ExperimentContext.Hyperparams


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


def transpose_conv2d(x, filters, kernel_size=5, strides=2, padding='same'):
    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    return tf.layers.conv2d_transpose(x, filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      kernel_initializer=kernel_initializer)


def batch_norm(x, training, epsilon=1e-5, momentum=0.9):
    training = True
    return tf.layers.batch_normalization(x, training=training, epsilon=epsilon, momentum=momentum)
    # return x

def dense(x, out_units, activation_fn=None):
    kernel = tf.random_normal_initializer(mean=0.0, stddev=0.3)
    #     return tf.layers.dense(x, out_units, activation=None)
    return layers.fully_connected(x, out_units, activation_fn=activation_fn)


def conv2d(x, filters, name, kernel_size=5, strides=2, padding="same", activation=None):
    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    return tf.layers.conv2d(x, filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                            kernel_initializer=kernel_initializer, name=name)


def encoder(x, alpha=0.2, training=True):  # change
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        # Input layer is 32x32x1
        conv1 = conv2d(x, 32, kernel_size=5, strides=1, name='conv1')
        conv1 = tf.nn.leaky_relu(conv1, alpha=alpha)
        print('conv1', conv1.shape)

        conv2 = conv2d(x, 64, kernel_size=4, strides=2, name='conv2')
        conv2 = batch_norm(conv2, training=training)
        conv2 = tf.nn.leaky_relu(conv2, alpha=alpha)
        print('conv2', conv2.shape)

        conv3 = conv2d(conv2, 128, kernel_size=4, strides=1, name='conv3')
        conv3 = batch_norm(conv3, training=training)
        conv3 = tf.nn.leaky_relu(conv3, alpha=alpha)
        print('conv3', conv3.shape)

        conv4 = conv2d(conv3, 256, kernel_size=4, strides=2, name='conv4')
        conv4 = batch_norm(conv4, training=training)
        conv4 = tf.nn.leaky_relu(conv4, alpha=alpha)
        print('conv4', conv4.shape)

        conv5 = conv2d(conv4, 512, kernel_size=4, strides=1, name='conv5')
        conv5 = batch_norm(conv5, training=training)
        conv5 = tf.nn.leaky_relu(conv5, alpha=alpha)
        print('conv5', conv5.shape)

        conv6 = conv2d(conv5, 512, kernel_size=4, strides=2, name='conv6')
        conv6 = batch_norm(conv6, training=training)
        conv6 = tf.nn.leaky_relu(conv6, alpha=alpha)
        print('conv6', conv6.shape)

        # Flatten it
        batch_size = tf.shape(x)[0]
        n_features = np.product(map(lambda x: x.value, conv6.shape[1:]))

        flat = tf.reshape(conv6, (batch_size, n_features))
        # print flat.shape
        z = dense(flat, H.z_size)
        print('z_recons  ', z.shape)

        return commons.scaled_tanh(z, H.z_bound)


def decoder(z, output_dim=1, training=True,alpha=0.2):  # change
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(z)[0]
        print "decoder"
        fc1 = dense(z, 4 * 4 * 256)
        fc1 = tf.reshape(fc1, (batch_size, 4, 4, 256))
        # fc1 = batch_norm(fc1, training=training)
        fc1 = tf.nn.leaky_relu(fc1)
        print fc1.shape

        conv1 = transpose_conv2d(fc1,filters=256 ,kernel_size=4,strides=2)
        conv1 = batch_norm(conv1, training=training)
        conv1 = tf.nn.leaky_relu(conv1,alpha=alpha)
        print conv1.shape
        conv2 = transpose_conv2d(conv1,filters=128 ,kernel_size=4,strides=2)
        conv2 = batch_norm(conv2, training=training)
        conv2 = tf.nn.leaky_relu(conv2, alpha=alpha)
        print conv2.shape
        conv3 = transpose_conv2d(conv2,filters=64 ,kernel_size=4,strides=2)
        conv3 = batch_norm(conv3, training=training)
        conv3 = tf.nn.leaky_relu(conv3, alpha=alpha)
        print conv3.shape
        conv4 = conv2d(conv3, filters=32,name = 'conv4_de', kernel_size=4, strides=1)
        conv4 = batch_norm(conv4, training=training)
        conv4 = tf.nn.leaky_relu(conv4, alpha=alpha)
        print conv4.shape
        conv5 = conv2d(conv4, filters=output_dim,name = 'conv5_de', kernel_size=5, strides=1)
        print conv5.shape
        gen_out = conv5
        print gen_out
        print ''
        print ''
        print ''
        out = tf.tanh(gen_out)
        return out  # 32x32x1


def disc(x, training=True,alpha=0.2):  # change
    with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
        # Input layer is 32x32x1
        conv1 = conv2d(x, 32, kernel_size=5, strides=1, name='conv1')
        conv1 = tf.nn.leaky_relu(conv1, alpha=alpha)
        print('conv1', conv1.shape)

        conv2 = conv2d(x, 64, kernel_size=4, strides=2, name='conv2')
        conv2 = batch_norm(conv2, training=training)
        conv2 = tf.nn.leaky_relu(conv2, alpha=alpha)
        print('conv2', conv2.shape)

        conv3 = conv2d(conv2, 128, kernel_size=4, strides=1, name='conv3')
        conv3 = batch_norm(conv3, training=training)
        conv3 = tf.nn.leaky_relu(conv3, alpha=alpha)
        print('conv3', conv3.shape)

        conv4 = conv2d(conv3, 256, kernel_size=4, strides=2, name='conv4')
        conv4 = batch_norm(conv4, training=training)
        conv4 = tf.nn.leaky_relu(conv4, alpha=alpha)
        print('conv4', conv4.shape)

        conv5 = conv2d(conv4, 512, kernel_size=4, strides=1, name='conv5')
        conv5 = batch_norm(conv5, training=training)
        conv5 = tf.nn.leaky_relu(conv5, alpha=alpha)
        print('conv5', conv5.shape)

        conv6 = conv2d(conv5, 512, kernel_size=4, strides=2, name='conv6')
        conv6 = batch_norm(conv6, training=training)
        conv6 = tf.nn.leaky_relu(conv6, alpha=alpha)
        print('conv6', conv6.shape)

        # Flatten it
        batch_size = tf.shape(x)[0]
        n_features = np.product(map(lambda x: x.value, conv6.shape[1:]))

        flat = tf.reshape(conv6, (batch_size, n_features))
        prelogits = dense(flat, 64)
        prelogits_fc = dense(prelogits, 16)
        # print flat.shape
        logits = dense(prelogits, 1)
        prelogits_reshape = tf.reshape(prelogits_fc, [-1, H.logit_batch_size * 16])
        entropy_logits = dense(prelogits_reshape, 1)
        prob = tf.sigmoid(logits)
        return prob, logits, entropy_logits
