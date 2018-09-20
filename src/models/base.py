import os
import logging
import paths
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorflow.contrib import framework as tf_framework

logger = logging.getLogger(__name__)


class BaseModel():
    __metaclass__ = ABCMeta

    def __init__(self, model_name, session=None):
        self.model_name = model_name
        self.model_scope = model_name
        # Weight Savers and Loaders
        self.param_groups = {}
        self.loggers = {}

        self.param_savers = {}
        self.weights_path = {}
        self.session = session

    def __repr__(self):
        return ('Model[%s]' % self.model_name)

    @abstractmethod
    def build(self):
        return NotImplemented

    @abstractmethod
    def initiate_service(self):
        """
        Override this function in the subclass and make sure to call this method from the overridden method.
        Summary Writers, Weight Savers will all be initiatized in this function after the build() function has been called.
        :return:
        """
        if self.session is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            self.session = tf.Session(config=config)

        init = tf.variables_initializer(tf.global_variables(self.model_name))
        self.session.run(init)
        local_params = tf.local_variables(self.model_name)
        local = tf.variables_initializer(local_params)
        self.session.run(local)
        uninit_var = tf.report_uninitialized_variables()
        uninit_var = self.session.run(uninit_var)

        if len(uninit_var) > 0:
            print 'uninit var: '
            for i in uninit_var:
                print i

        for name in ['train', 'test']:
            self.add_logger(name, paths.log_writer_path(name))

    @property
    def network_names(self):
        return self.param_groups.keys()

    def add_param_saver(self, name, params, max_to_keep=3):
        if name in self.param_savers:
            logger.error('Duplicate Saver Error: Trying to add duplicate saver %s to model %s'.format(name, self.model_name))
            raise Exception('Saver already exists...')
        saver = tf.train.Saver(params, max_to_keep=max_to_keep)
        self.param_savers[name] = saver

    def get_param_saver(self, name):
        param_saver = self.param_savers[name]  # type: tf.train.Saver
        return param_saver

    def add_logger(self, name, log_path, graph=None):
        if name in self.loggers:
            logger.error('Duplicate Logger Error: Trying to add duplicate logger %s to model %s'.format(name, self.model_name))
            raise Exception('Logger already exists...')
        summary_logger = tf.summary.FileWriter(log_path, graph, flush_secs=60)
        self.loggers[name] = summary_logger

    def get_logger(self, name):
        summary_logger = self.loggers[name]  # type: tf.summary.FileWriter
        return summary_logger

    def add_param_group(self, name, param_group):
        if name in self.param_groups:
            logger.error('Duplicate Params Error: Trying to add duplicate params %s to model %s'.format(name, self.model_name))
            raise Exception('Params already exists...')
        self.param_groups[name] = param_group

    def save_params(self, dir_name='all', param_group='all', tag='iter', iter_no=None):
        """
        :param dir_name: Directory inside which weights need to be stored
        :param param_group: parameter group to select the writer.
        :param tag: additional tag to be added to the suffix of the name of the weights, if required. Default: ''
        :param iter_no: additional iteration_no to be added in the end of the name, if required. Default: NO_ITERATION
        :return:
        """
        weights_path = paths.get_saved_params_path(dir_name, self.model_name, param_group, tag, iter_no=None)
        logger.info('Saving weights at %s: iter_no: %d' % (weights_path, iter_no))
        self.param_savers[param_group].save(self.session, weights_path, global_step=iter_no)

    def load_params_from_model(self, parent_model):
        parent_model_param_values = parent_model.get_all_param_values()
        all_new_variable_names = map(lambda v: v.name[:-2], tf.trainable_variables(self.model_name))

        with tf.variable_scope("", reuse=True):
            for parent_var_name in parent_model_param_values:
                new_variable_name = parent_var_name.replace(parent_model.model_name, self.model_name)
                if new_variable_name not in all_new_variable_names:
                    print 'Oops!, {} not in var list'.format(new_variable_name)
                var_value = parent_model_param_values[parent_var_name]
                new_variable = tf.get_variable(new_variable_name)
                new_variable.load(var_value, self.session)

    def load_params_from_history(self, dir_name='all', param_group='all', tag='iter', iter_no=None):
        param_saver = self.get_param_saver(param_group)
        weights_path = paths.get_saved_params_path(dir_name, self.model_name, param_group, tag, iter_no)
        if iter_no is None:
            checkpoint_dir = os.path.dirname(weights_path)
            weights_path = tf.train.latest_checkpoint(checkpoint_dir)
            iter_no = int(weights_path.split('-')[-1])
        param_saver.restore(self.session, weights_path)
        return iter_no

    def load_params(self, dir_name='all', param_group='all', tag='iter', iter_no=None):
        """
        Will load the latest params, if iter_no is None.
        """
        weights_path = paths.get_saved_params_path(dir_name, self.model_name, param_group, tag, iter_no)
        if iter_no is None:
            checkpoint_dir = os.path.dirname(weights_path)
            weights_path = tf.train.latest_checkpoint(checkpoint_dir)
            iter_no = int(weights_path.split('-')[-1])

        old_variable_names = tf_framework.list_variables(weights_path)
        all_new_variable_names = map(lambda v: v.name, tf.global_variables(self.model_name))
        print('All Variable names:\n')
        # for var_name in all_new_variable_names:
        #     print var_name
        with tf.variable_scope("", reuse=True):
            for old_var_name, _ in old_variable_names:
                old_model_scope = old_var_name.split('/')[0]
                new_variable_name = old_var_name.replace(old_model_scope, self.model_name)
                if new_variable_name + ':0' not in all_new_variable_names:
                    print 'Oops!, {} not in var list'.format(new_variable_name)
                print(new_variable_name)
                if 'RMSProp' in new_variable_name:
                    print 'ignoreing...'
                    continue
                var_value = tf_framework.load_variable(weights_path, old_var_name)
                new_variable = tf.get_variable(new_variable_name)
                new_variable.load(var_value, self.session)
        return iter_no

    def log_custom_summary(self, logger_name, tag, simple_value, iter_no):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=tag, simple_value=simple_value),
        ])
        self.loggers[logger_name].add_summary(summary, global_step=iter_no)

    def get_all_param_values(self):
        param_values = {}
        params = tf.trainable_variables(self.model_name)
        values = self.session.run(params)
        for param, value in zip(params, values):
            param_values[param.name[:-2]] = value
        return param_values

    @abstractmethod
    def encode(self, x, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def decode(self, z, *args, **kwargs):
        return NotImplemented
