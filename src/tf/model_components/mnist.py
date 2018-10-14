import tensorflow as tf
from tensorflow.contrib import layers
from exp_context import ExperimentContext

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


def lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


def transpose_conv2d(x, filters):
    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    return tf.layers.conv2d_transpose(x, filters, 5, strides=2, padding='same', kernel_initializer=kernel_initializer)


def batch_norm(x, training, epsilon=1e-5, momentum=0.9):
    return tf.layers.batch_normalization(x, training=training, epsilon=epsilon, momentum=momentum)


def dense(x, out_units):
    kernel = tf.random_normal_initializer(mean=0.0, stddev=0.3)
    #     return tf.layers.dense(x, out_units, activation=None)
    return layers.fully_connected(x, out_units, activation_fn=None)


def conv2d(x, filters, name):
    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    return tf.layers.conv2d(x, filters, kernel_size=5, strides=2, padding="same", activation=None,
                            kernel_initializer=kernel_initializer, name=name)


def encoder(x, alpha=0.2, training=True):  # change
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        # Input layer is 32x32x1
        conv1 = conv2d(x, 64, name='conv1')
        conv1 = tf.nn.leaky_relu(conv1, alpha)
        print conv1.shape

        conv2 = conv2d(conv1, 128, name='conv2')
        conv2 = batch_norm(conv2, training=training)
        conv2 = tf.nn.leaky_relu(conv2, alpha)
        print conv2.shape

        conv3 = conv2d(conv2, 256, name='conv3')
        conv3 = batch_norm(conv3, training=training)
        conv3 = tf.nn.leaky_relu(conv3, alpha)
        print conv3.shape
        # Flatten it
        flat = tf.reshape(conv3, (-1, 4 * 4 * 256))
        # print flat.shape
        z_reconstruct = dense(flat, 100)
        print 'z_recons  ', z_reconstruct.shape
        logits = dense(flat, 1)
        logits_reshape = tf.reshape(logits, [-1, H.logit_batch_size])
        entropy_logits = dense(logits_reshape, 1)
        out = tf.sigmoid(logits)

        return out, logits, entropy_logits, z_reconstruct


def decoder(z, output_dim=1, training=True):  # change
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        fc1 = dense(z, 4 * 4 * 256)

        fc1 = tf.reshape(fc1, (-1, 4, 4, 256))
        fc1 = batch_norm(fc1, training=training)
        fc1 = tf.nn.relu(fc1)
        # 4x4
        print fc1.shape
        t_conv1 = transpose_conv2d(fc1, 128)
        t_conv1 = batch_norm(t_conv1, training=training)
        t_conv1 = tf.nn.relu(t_conv1)
        # 8x8
        print t_conv1.shape

        t_conv2 = transpose_conv2d(t_conv1, 64)
        t_conv2 = batch_norm(t_conv2, training=training)
        t_conv2 = tf.nn.relu(t_conv2)
        # 16x16
        print t_conv2.shape

        t_conv3 = transpose_conv2d(t_conv2, output_dim)
        # 32x32
        gen_out = t_conv3
        print gen_out

        out = tf.tanh(gen_out)
        return gen_out  # 32x32x1


def disc(x, training=True):  # change
    out, logits, z = encoder(x, training=training)
    return out, logits, z
