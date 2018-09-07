# Autoencoder
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.contrib import layers
import data

learning_rate = 0.0001

with tf.variable_scope('gan', reuse=tf.AUTO_REUSE):
    num_input = 2
    X = tf.placeholder("float", [None, num_input])
    Z = tf.placeholder("float", [None, 1])

    batch_size = tf.shape(X)[0]


    def encoder(x):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            l1 = layers.fully_connected(x, 128, scope='l1', reuse=tf.AUTO_REUSE)
            l2 = layers.fully_connected(l1, 64, scope='l2', reuse=tf.AUTO_REUSE)
            l4 = layers.fully_connected(l2, 1, activation_fn=tf.nn.leaky_relu, scope='l4', reuse=tf.AUTO_REUSE)
            return l4


    def decoder(x):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            l1 = layers.fully_connected(x, 64, scope='l1', reuse=tf.AUTO_REUSE)
            l2 = layers.fully_connected(l1, 128, scope='l2', reuse=tf.AUTO_REUSE)
            l3 = layers.fully_connected(l2, 2, activation_fn=tf.nn.leaky_relu, scope='l3', reuse=tf.AUTO_REUSE)
            return l3


    def disc(x):
        with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
            l1 = layers.fully_connected(x, 128, scope='l1', reuse=tf.AUTO_REUSE)
            l2 = layers.fully_connected(l1, 128, scope='l2', reuse=tf.AUTO_REUSE)
            l3 = layers.fully_connected(l2, 128, scope='l3', reuse=tf.AUTO_REUSE)
            l5 = layers.fully_connected(l3, 128, scope='l5', reuse=tf.AUTO_REUSE)
            logits = layers.fully_connected(l5, 2, activation_fn=tf.nn.leaky_relu, scope='logits', reuse=tf.AUTO_REUSE)

            real_labels = tf.concat([tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])], axis=1)
            fake_labels = tf.concat([tf.zeros([batch_size, 1]), tf.ones([batch_size, 1])], axis=1)

            real_loss = tf.losses.softmax_cross_entropy(real_labels, logits)
            fake_loss = tf.losses.softmax_cross_entropy(fake_labels, logits)

            pred = tf.nn.softmax(logits)

            real_acc = tf.reduce_mean(tf.cast(pred[:, 0] >= 0.5, tf.float32))
            fake_acc = tf.reduce_mean(tf.cast(pred[:, 1] >= 0.5, tf.float32))

            return {
                'real_loss': real_loss,
                'fake_loss': fake_loss,
                'real_acc': real_acc,
                'fake_acc': fake_acc,

            }

x_real = X
z_real = encoder(x_real)
x_recon = decoder(z_real)

z_rand = Z
x_fake = decoder(z_rand)
z_recon = encoder(x_fake)

x_recon_loss = tf.reduce_mean((x_real - x_recon) ** 2)
z_recon_loss = tf.reduce_mean((z_rand - z_recon) ** 2)

disc_real = disc(x_real)
disc_fake = disc(x_fake)

disc_loss = tf.reduce_mean(disc_real['real_loss'] + disc_fake['fake_loss'])
gen_loss = tf.reduce_mean(disc_fake['real_loss'])

disc_real_acc = 100 * disc_real['real_acc']
disc_fake_acc = 100 * disc_fake['fake_acc']

disc_acc = 0.5 * (disc_real_acc + disc_fake_acc)
gen_acc = 100 * (disc_fake['real_acc'])

encoder_loss = x_recon_loss + z_recon_loss
decoder_loss = encoder_loss + gen_loss

encoder_train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(encoder_loss)
gen_train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(gen_loss)
disc_train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(disc_loss)
decoder_train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(decoder_loss)

summaries_list = [
    tf.summary.scalar('x_recon_loss', x_recon_loss),
    tf.summary.scalar('z_recon_loss', z_recon_loss),
    tf.summary.scalar('encoder_loss', encoder_loss),
    tf.summary.scalar('gen_loss', z_recon_loss),
    tf.summary.scalar('disc_loss', disc_loss),
    tf.summary.scalar('gen_acc', gen_acc),
    tf.summary.scalar('disc_acc', disc_acc),
    tf.summary.scalar('disc_real_acc', disc_real_acc),
    tf.summary.scalar('disc_fake_acc', disc_fake_acc),

    tf.summary.histogram('z_rand', z_rand),
    tf.summary.histogram('z_real', z_real),
    tf.summary.histogram('z_recon', z_recon),
]

summaries = tf.summary.merge(summaries_list)
