# Autoencoder
import logging
import numpy as np
import tensorflow as tf

from model_components import losses
from model_components.toy import encoder, decoder, disc_v2
from models_tf.base import BaseModel
from exp_context import ExperimentContext

H = ExperimentContext.Hyperparams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Model(BaseModel):

    def __init__(self, model_name, session=None):
        BaseModel.__init__(self, model_name, session)
        self.__is_model_built = False

    def build(self):
        self._define_placeholders()
        self._define_network_graph()
        self._define_losses()
        self._define_metrics()
        self._define_summaries()
        self._define_scopes()
        self._define_operations()
        self._create_param_groups()

        logger.info('%s: Model Definition complete' % repr(self))

        # logger.info('%s: Model Params:' % repr(self))
        # for param in tf.trainable_variables(self.model_scope):
        #     logger.info(param)

        self.__is_model_built = True

    def initiate_service(self):
        if not self.__is_model_built:
            logger.error('Model Service Initiation Error: Trying to initiate service before building the model.')
            logger.error('execute model.build() before model.initiate_service()')
            raise Exception('Model Service Initiation Error: Trying to initiate service before building the model.')
        BaseModel.initiate_service(self)

        for network_name, param_list in self.param_groups.items():
            self.add_param_saver(network_name, param_list)

    def _define_placeholders(self):
        self.ph_X = tf.placeholder(tf.float32, [None, H.input_size])
        self.ph_Z = tf.placeholder(tf.float32, [None, H.z_size])
        self.ph_plot_img = tf.placeholder(tf.float32, [None, None, None, 4])

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
            self.logits_real, self.entropy_logits_real = disc_v2(self.x_real)
            self.logits_fake, self.entropy_logits_fake = disc_v2(self.x_fake)

    def _define_losses(self):
        batch_size = tf.shape(self.ph_X)[0]
        logit_batch_size = tf.shape(self.entropy_logits_real)[0]

        self.x_recon_loss = tf.reduce_mean((self.x_real - self.x_recon) ** 2)
        self.z_recon_loss = tf.reduce_mean((self.z_rand - self.z_recon) ** 2)
        # [B, 2]

        real_entropy_labels = tf.ones([logit_batch_size, 1])
        fake_entropy_labels = tf.zeros([logit_batch_size, 1])

        real_labels = tf.ones([batch_size, 1])
        fake_labels = tf.zeros([batch_size, 1])

        self.disc_batch_loss_real = losses.sigmoid_cross_entropy_loss(real_entropy_labels, self.entropy_logits_real)
        self.disc_batch_loss_fake = losses.sigmoid_cross_entropy_loss(fake_entropy_labels, self.entropy_logits_fake)

        self.gen_batch_loss = losses.sigmoid_cross_entropy_loss(real_entropy_labels, self.entropy_logits_fake)

        self.disc_sample_loss_real = losses.sigmoid_cross_entropy_loss(real_labels, self.logits_real)
        self.disc_sample_loss_fake = losses.sigmoid_cross_entropy_loss(fake_labels, self.logits_fake)

        self.gen_sample_loss = losses.sigmoid_cross_entropy_loss(real_labels, self.logits_fake)

        if H.train_sample_logits and H.train_batch_logits:
            self.disc_loss_real = (self.disc_batch_loss_real + self.disc_sample_loss_real) / 2.0
            self.disc_loss_fake = (self.disc_batch_loss_fake + self.disc_sample_loss_fake) / 2.0
            self.gen_loss = (self.gen_batch_loss + self.gen_sample_loss) / 2.0

        elif H.train_sample_logits:
            self.disc_loss_real = self.disc_sample_loss_real
            self.disc_loss_fake = self.disc_sample_loss_fake
            self.gen_loss = self.gen_sample_loss

        elif H.train_batch_logits:
            self.disc_loss_real = self.disc_batch_loss_real
            self.disc_loss_fake = self.disc_batch_loss_fake
            self.gen_loss = self.gen_batch_loss

        else:
            logger.error('Logits Training set False for both sample and batch: Atleast one must be set True in Hyperparams')
            raise Exception('Logits not set to train')

        self.encoder_loss = self.x_recon_loss + self.z_recon_loss
        self.decoder_loss = self.encoder_loss + 0 * self.gen_loss
        self.disc_loss = self.disc_loss_real + self.disc_loss_fake

    def _define_metrics(self):
        self.disc_real_preds = tf.cast(self.logits_real >= 0., tf.int32)
        self.disc_fake_preds = tf.cast(self.logits_fake >= 0., tf.int32)

        self.disc_real_acc = 100 * tf.reduce_mean(tf.cast(tf.equal(self.disc_real_preds, 1), H.dtype))
        self.disc_fake_acc = 100 * tf.reduce_mean(tf.cast(tf.equal(self.disc_fake_preds, 0), H.dtype))

        self.disc_acc = 0.5 * (self.disc_real_acc + self.disc_fake_acc)
        self.gen_acc = 100 - self.disc_fake_acc

    def _define_scopes(self):
        self.encoder_scope = self.model_scope + '/encoder'
        self.decoder_scope = self.model_scope + '/decoder'
        self.disc_scope = self.model_scope + '/disc'

    def _define_summaries(self):
        summaries_list = [
            tf.summary.scalar('recon_x_loss', self.x_recon_loss),
            tf.summary.scalar('recon_z_loss', self.z_recon_loss),
            tf.summary.scalar('recon_loss', self.encoder_loss),

            tf.summary.scalar('gen_loss', self.gen_loss),
            tf.summary.scalar('gen_loss_batch', self.gen_batch_loss),
            tf.summary.scalar('gen_loss_samples', self.gen_sample_loss),

            tf.summary.scalar('disc_loss', self.disc_loss),

            tf.summary.scalar('disc_loss_sample_real', self.disc_sample_loss_real),
            tf.summary.scalar('disc_loss_sample_fake', self.disc_sample_loss_fake),

            tf.summary.scalar('disc_loss_batch_real', self.disc_batch_loss_real),
            tf.summary.scalar('disc_loss_batch_fake', self.disc_batch_loss_fake),

            tf.summary.scalar('disc_loss_real', self.disc_loss_real),
            tf.summary.scalar('disc_loss_fake', self.disc_loss_fake),

            tf.summary.scalar('gen_acc', self.gen_acc),

            tf.summary.scalar('disc_acc', self.disc_acc),
            tf.summary.scalar('disc_acc_real', self.disc_real_acc),
            tf.summary.scalar('disc_acc_fake', self.disc_fake_acc),

            tf.summary.histogram('z_rand', self.z_rand),
            tf.summary.histogram('z_real', self.z_real),
            tf.summary.histogram('z_recon', self.z_recon),

            tf.summary.image("Generated images",self.x_fake)
            # tf.summary.image("")

            # tf.summary.histogram('fake_preds', self.disc_real_preds),
            # tf.summary.histogram('real_preds', self.disc_fake_preds),
        ]

        self.summaries = tf.summary.merge(summaries_list)
        self.img_summary = tf.summary.image('viz_plots', self.ph_plot_img)

    def _create_param_groups(self):
        self.add_param_group('encoder', tf.global_variables(self.model_scope + '/encoder'))
        self.add_param_group('decoder', tf.global_variables(self.model_scope + '/decoder'))
        self.add_param_group('autoencoder', self.param_groups['encoder'] + self.param_groups['decoder'])
        self.add_param_group('disc', tf.global_variables(self.model_scope + '/disc'))
        self.add_param_group('all', tf.global_variables(self.model_scope))

    def _define_operations(self):
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            autoencoder_trainable_params = tf.trainable_variables(self.encoder_scope) + tf.trainable_variables(self.decoder_scope)
            opt = tf.train.RMSPropOptimizer(H.lr_autoencoder)
            self.autoencoder_train_op = opt.minimize(self.encoder_loss, var_list=autoencoder_trainable_params)

            decoder_params = tf.trainable_variables(self.decoder_scope)
            opt = tf.train.RMSPropOptimizer(H.lr_autoencoder)
            self.adv_gen_train_op = opt.minimize(self.gen_loss, var_list=decoder_params)

            disc_params = tf.trainable_variables(self.disc_scope)
            opt = tf.train.RMSPropOptimizer(H.lr_autoencoder)
            self.adv_disc_train_op = opt.minimize(self.disc_loss, var_list=disc_params)

    @property
    def network_loss_variables(self):
        network_losses = [
            self.encoder_loss,
            self.disc_acc,
            self.gen_acc,
            self.x_recon_loss,
            self.z_recon_loss,
            self.summaries
        ]
        return network_losses

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
        })[:, 0]
        if split:
            x_batch_real = x_batch[np.where(preds == 1)]
            x_batch_fake = x_batch[np.where(preds == 0)]
            return preds, x_batch_real, x_batch_fake
        else:
            return preds

    def log_image(self, logger_name, image, iter_no):
        summary = self.session.run(self.img_summary, feed_dict={
            self.ph_plot_img: image[None, :, :, :]
        })
        self.loggers[logger_name].add_summary(summary, global_step=iter_no)


if __name__ == '__main__':
    model = Model('ggans')
    model.build()
    model.initiate_service()
