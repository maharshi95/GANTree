import tensorflow as tf
from exp_context import ExperimentContext
from . import commons

H = ExperimentContext.Hyperparams


def encoder(x):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        n_units = [128, 128, 64, 64, H.z_size]
        n_layers = len(n_units)
        activations = [tf.nn.elu] * (n_layers - 1) + [commons.get_scaled_tanh(4.0)]
        z = commons.n_layers_dense(x, n_units, activations)
        return z


def decoder(z):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        n_units = [64, 64, 128, 128, H.input_size]
        n_layers = len(n_units)
        activations = [tf.nn.elu] * (n_layers - 1) + [None]
        x = commons.n_layers_dense(z, n_units, activations)
        return x


def disc(x):
    with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
        n_units = [64, 64, 128, 128, 1]
        n_layers = len(n_units)
        act_fn = [tf.nn.elu] * (n_layers - 1) + [None]
        logits = commons.n_layers_dense(x, n_units, act_fn)
        return logits


def disc_v2(x):
    with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
        n_units = [64, 64, 128, 32]
        n_layers = len(n_units)
        act_fn = [tf.nn.elu] * n_layers
        pre_logits = commons.n_layers_dense(x, n_units, act_fn, name='dense_1')

        sample_logits = commons.dense(pre_logits, 1, None)

        pre_logits_reshaped = tf.reshape(pre_logits, [-1, H.logit_batch_size * n_units[-1]])
        n_units_2 = [64, 32, 1]
        act_fn_2 = [tf.nn.elu, tf.nn.elu, None]
        entropy_logits = commons.n_layers_dense(pre_logits_reshaped, n_units_2, act_fn_2, name='dense_2')

        return sample_logits, entropy_logits
