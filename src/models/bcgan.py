# Autoencoder
import logging
import numpy as np
import tensorflow as tf

from model_components import encoder, decoder, disc
from models.base import BaseModel
from hyperparams.base import Hyperparams as H

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Model(BaseModel):

    def __init__(self, model_name):
        BaseModel.__init__(self, model_name)
        self.model_scope = 'growing_gans'
        self.__is_model_build = False

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
        for param in tf.trainable_variables(self.model_scope):
            logger.info(param)

        self.__is_model_build = True

    def initiate_service(self):
        if not self.__is_model_build:
            logger.error('Model Service Initiation Error: Trying to initiate service before building the model.')
            logger.error('execute model.build() before model.initiate_service()')
            raise Exception('Model Service Initiation Error: Trying to initiate service before building the model.')
        BaseModel.initiate_service(self)

        for network_name, param_list in self.param_groups.items():
            self.add_param_saver(network_name, param_list)

    def _define_placeholders(self):
        self.ph_X = tf.placeholder(tf.float32, [None, H.input_size])
        self.ph_Z = tf.placeholder(tf.float32, [None, H.z_size])

    def _define_network_graph(self):
        with tf.variable_scope(self.model_scope, reuse=tf.AUTO_REUSE):
            # X - Z - X Iteration
            self.x_real = self.ph_X
            self.z_real = encoder(self.x_real)
            self.x_recon = decoder(self.z_real)

            # Z - X - Z Iteration
            self.z_rand = self.ph_Z
            self.x_fake = decoder(self.z_rand)
            self.z_recon = encoder(self.x_fake)

            # Disc Iteration
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

        self.disc_real_acc = 100 * tf.reduce_mean(tf.cast(tf.equal(self.disc_real_preds, 0), H.dtype))
        self.disc_fake_acc = 100 * tf.reduce_mean(tf.cast(tf.equal(self.disc_fake_preds, 1), H.dtype))

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

            tf.summary.histogram('fake_preds', self.disc_real_preds),
            tf.summary.histogram('real_preds', self.disc_fake_preds),
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

        decoder_params = tf.trainable_variables(self.decoder_scope)
        self.adv_gen_train_op = tf.train.RMSPropOptimizer(H.lr_decoder) \
            .minimize(self.gen_loss, var_list=decoder_params)

        disc_params = tf.trainable_variables(self.disc_scope)
        self.adv_disc_train_op = tf.train.RMSPropOptimizer(H.lr_disc) \
            .minimize(self.disc_loss, var_list=disc_params)

    def step_train_autoencoder(self, inputs):
        x_input, z_input = inputs
        self.session.run(self.autoencoder_train_op, feed_dict={
            self.ph_X: x_input,
            self.ph_Z: z_input,
        })

    def step_train_adv_generator(self, inputs):
        x_input, z_input = inputs
        self.session.run(self.adv_gen_train_op, feed_dict={
            self.ph_X: x_input,
            self.ph_Z: z_input,
        })

    def step_train_discriminator(self, inputs):
        x_input, z_input = inputs
        self.session.run(self.adv_disc_train_op, feed_dict={
            self.ph_X: x_input,
            self.ph_Z: z_input,
        })

    def compute_losses(self, inputs, losses):
        x_input, z_input = inputs
        network_outputs = self.session.run(losses, feed_dict={
            self.ph_X: x_input,
            self.ph_Z: z_input,
        })
        return network_outputs

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return self.session.run(fetches, feed_dict, options, run_metadata)

    def encode(self, x_batch):
        return self.session.run(self.z_real, {
            self.ph_X: x_batch
        })

    def decode(self, z_batch):
        return self.session.run(self.x_fake, {
            self.ph_Z: z_batch
        })

    def reconstruct_x(self, x_batch):
        return self.session.run(self.x_recon, {
            self.ph_X: x_batch
        })

    def reconstruct_z(self, z_batch):
        return self.session.run(self.z_recon, {
            self.ph_Z: z_batch
        })

    def discriminate(self, x_batch, split=True):
        preds = self.session.run(self.disc_real_preds, {
            self.ph_X: x_batch
        })
        if split:
            x_batch_real = x_batch[np.where(preds == 0)]
            x_batch_fake = x_batch[np.where(preds == 1)]
            return preds, x_batch_real, x_batch_fake
        else:
            return preds


if __name__ == '__main__':
    model = Model('ggans')
    model.build()
    model.initiate_service()
