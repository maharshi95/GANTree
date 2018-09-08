# Autoencoder
import os, logging
import tensorflow as tf
from models.base import BaseModel
from hyperparams.base import Hyperparams as H
from tensorflow.contrib import layers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encoder(x):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        fc1 = layers.fully_connected(x, 128, scope='fc1', reuse=tf.AUTO_REUSE)
        fc2 = layers.fully_connected(fc1, 64, scope='fc2', reuse=tf.AUTO_REUSE)
        fc3 = layers.fully_connected(fc2, 1, activation_fn=None, scope='fc3', reuse=tf.AUTO_REUSE)
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
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        fc1 = layers.fully_connected(x, 128, scope='fc1', reuse=tf.AUTO_REUSE)
        fc2 = layers.fully_connected(fc1, 128, scope='fc2', reuse=tf.AUTO_REUSE)
        fc3 = layers.fully_connected(fc2, 128, scope='fc3', reuse=tf.AUTO_REUSE)
        fc4 = layers.fully_connected(fc3, 128, scope='fc4', reuse=tf.AUTO_REUSE)
        # [B, 2]
        logits = layers.fully_connected(fc4, 2, activation_fn=None, scope='logits', reuse=tf.AUTO_REUSE)

        return logits


class Model(BaseModel):

    def __init__(self, model_name):
        BaseModel.__init__(self, model_name)
        self.model_scope = 'growing_gans'

    def initiate_service(self):
        BaseModel.initiate_service(self)


    def build(self):
        self._define_placeholders()
        self._define_network_graph()
        self._define_losses()
        self._define_metrics()
        self._define_summaries()
        self._define_scopes()
        self._define_operations()
        self._create_param_groups()

        logger.info('Model Definition complete')
        logger.info('Model Params:')
        for param in self.param_groups['all']:
            logger.info(param.name)

    def _define_placeholders(self):
        self.ph_X = tf.placeholder(tf.float32, [None, H.input_size])
        self.ph_Z = tf.placeholder(tf.float32, [None, H.z_size])

    def _define_network_graph(self):
        with tf.variable_scope(self.model_scope, reuse=tf.AUTO_REUSE):
            self.x_real = self.ph_X
            self.z_real = encoder(self.x_real)
            self.x_recon = decoder(self.z_real)

            self.z_rand = self.ph_Z
            self.x_fake = decoder(self.z_rand)
            self.z_recon = encoder(self.x_fake)

            self.logits_real = disc(self.x_real)
            self.logits_fake = disc(self.x_fake)

    def _define_losses(self):
        batch_size = tf.shape(self.ph_X)[0]
        self.x_recon_loss = tf.reduce_mean((self.x_real - self.x_recon) ** 2)
        self.z_recon_loss = tf.reduce_mean((self.z_rand - self.z_recon) ** 2)
        # [B, 2]
        real_labels = tf.concat([tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])], axis=1)
        fake_labels = tf.concat([tf.zeros([batch_size, 1]), tf.ones([batch_size, 1])], axis=1)

        self.disc_loss_real = tf.losses.softmax_cross_entropy(real_labels, self.logits_real)
        self.disc_loss_fake = tf.losses.softmax_cross_entropy(fake_labels, self.logits_fake)

        self.gen_loss = tf.losses.softmax_cross_entropy(real_labels, self.logits_fake)

        self.encoder_loss = self.x_recon_loss + self.z_recon_loss
        self.decoder_loss = self.encoder_loss + 0 * self.gen_loss
        self.disc_loss = self.disc_loss_real + self.disc_loss_fake

    def _define_metrics(self):
        self.disc_real_preds = tf.argmax(self.logits_real, axis=-1)
        self.disc_fake_preds = tf.argmax(self.logits_fake, axis=-1)

        self.disc_real_acc = 100 * tf.reduce_mean(tf.cast(self.disc_real_preds == 0., H.dtype))
        self.disc_fake_acc = 100 * tf.reduce_mean(tf.cast(self.disc_fake_preds == 1., H.dtype))

        self.disc_acc = 0.5 * (self.disc_real_acc + self.disc_fake_acc)
        self.gen_acc = 100 - self.disc_fake_acc

    def _define_scopes(self):
        self.encoder_scope = self.model_scope + '/encoder'
        self.decoder_scope = self.model_scope + '/decoder'
        self.disc_scope = self.model_scope + '/disc'

    def _define_summaries(self):
        summaries_list = [
            tf.summary.scalar('x_recon_loss', self.x_recon_loss),
            tf.summary.scalar('z_recon_loss', self.z_recon_loss),
            tf.summary.scalar('encoder_loss', self.encoder_loss),
            tf.summary.scalar('gen_loss', self.z_recon_loss),
            tf.summary.scalar('disc_loss', self.disc_loss),
            tf.summary.scalar('gen_acc', self.gen_acc),
            tf.summary.scalar('disc_acc', self.disc_acc),
            tf.summary.scalar('disc_real_acc', self.disc_real_acc),
            tf.summary.scalar('disc_fake_acc', self.disc_fake_acc),

            tf.summary.histogram('z_rand', self.z_rand),
            tf.summary.histogram('z_real', self.z_real),
            tf.summary.histogram('z_recon', self.z_recon),
        ]

        self.summaries = tf.summary.merge(summaries_list)

    def _create_param_groups(self):
        self.add_param_group('encoder', tf.global_variables(self.model_scope + '/encoder'))
        self.add_param_group('decoder', tf.global_variables(self.model_scope + '/decoder'))
        self.add_param_group('autoencoder', self.param_groups['encoder'] + self.param_groups['decoder'])
        self.add_param_group('disc', tf.global_variables(self.model_scope + '/disc'))
        self.add_param_group('all', tf.global_variables(self.model_scope))

    def _define_operations(self):
        autoencoder_trainable_params = tf.trainable_variables(self.encoder_scope) + tf.trainable_variables(self.decoder_scope)
        self.autoencoder_train_op = tf.train.RMSPropOptimizer(H.lr_autoencoder) \
            .minimize(self.encoder_loss, var_list=autoencoder_trainable_params)

        self.adv_gen_train_op = tf.train.RMSPropOptimizer(H.lr_decoder) \
            .minimize(self.gen_loss, var_list=tf.trainable_variables(self.decoder_scope))

        self.adv_disc_train_op = tf.train.RMSPropOptimizer(H.lr_disc) \
            .minimize(self.disc_loss, var_list=tf.trainable_variables(self.disc_scope))

    def step_train_autoencoder(self, inputs):
        x_train, z_train = inputs
        self.session.run(self.autoencoder_train_op, feed_dict={
            self.ph_X: x_train,
            self.ph_Z: z_train,
        })

    def step_train_adv_generator(self, inputs):
        x_train, z_train = inputs
        self.session.run(self.adv_gen_train_op, feed_dict={
            self.ph_X: x_train,
            self.ph_Z: z_train,
        })

    def step_train_discriminator(self, inputs):
        x_train, z_train = inputs
        self.session.run(self.adv_disc_train_op, feed_dict={
            self.ph_X: x_train,
            self.ph_Z: z_train,
        })

    def compute_losses(self, inputs, losses):
        x_train, z_train = inputs
        network_outputs = self.session.run(losses, feed_dict={
            self.ph_X: x_train,
            self.ph_Z: z_train,
        })
        return network_outputs

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return self.session.run(fetches, feed_dict, options, run_metadata)


if __name__ == '__main__':
    model = Model('ggans')
    model.build()
    model.initiate_service()
