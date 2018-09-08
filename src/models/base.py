import os
import logging
import paths
from abc import ABCMeta, abstractmethod
import tensorflow as tf

logger = logging.getLogger(__name__)


class BaseModel():
    __metaclass__ = ABCMeta

    def __init__(self, model_name):
        self.model_name = model_name
        # Weight Savers and Loaders
        self.param_groups = {}
        self.loggers = {}

        self.param_savers = {}
        self.weights_path = {}

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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.InteractiveSession(config=config)
        self.session.run(tf.global_variables_initializer())

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
        weights_path = paths.get_saved_params_path(dir_name, param_group, tag, iter_no=None)
        self.param_savers[param_group].save(self.session, weights_path, global_step=iter_no)

    def load_params(self, dir_name='all', param_group='all', tag='iter', iter_no=None):
        """
        Will load the latest params, if iter_no is None.
        """
        param_saver = self.get_param_saver(param_group)
        weights_path = paths.get_saved_params_path(dir_name, param_group, tag, iter_no)
        if iter_no is None:
            checkpoint_dir = os.path.dirname(weights_path)
            full_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
            iter_no = int(full_checkpoint_path.split('-')[-1])
            param_saver.restore(self.session, full_checkpoint_path)

        else:
            param_saver.restore(self.session, weights_path)
        return iter_no

    def log_custom_summary(self, logger_name, tag, simple_value, iter_no):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=tag, simple_value=simple_value),
        ])
        self.loggers[logger_name].add_summary(summary, global_step=iter_no)
