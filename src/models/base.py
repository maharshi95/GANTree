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

        self.weight_savers = {}
        self.weights_path = {}

    @abstractmethod
    def build(self):
        return NotImplemented

    @abstractmethod
    def initiate_service(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.InteractiveSession(config=config)
        self.session.run(tf.global_variables_initializer())

    @property
    def network_names(self):
        return self.param_groups.keys()

    def add_saver(self, name, params, max_to_keep=3):
        if name in self.weight_savers:
            logger.error('Duplicate Saver Error: Trying to add duplicate saver %s to model %s'.format(name, self.model_name))
            raise Exception('Saver already exists...')
        saver = tf.train.Saver(params, max_to_keep=max_to_keep)
        self.weight_savers[name] = saver

    def add_summary_logger(self, name, log_path, graph=None):
        if name in self.loggers:
            logger.error('Duplicate Logger Error: Trying to add duplicate logger %s to model %s'.format(name, self.model_name))
            raise Exception('Logger already exists...')
        summary_logger = tf.summary.FileWriter(log_path, graph, flush_secs=60)
        self.loggers[name] = summary_logger

    def add_param_group(self, name, param_group):
        if name in self.param_groups:
            logger.error('Duplicate Params Error: Trying to add duplicate params %s to model %s'.format(name, self.model_name))
            raise Exception('Params already exists...')
        self.param_groups[name] = param_group

    def save_params(self, dir_name, param_group=None, tag='', iter_no=None):
        weights_path = paths.get_saved_params_path(dir_name, param_group, tag, iter_no)
        self.weight_savers[param_group].save(self.session, weights_path, global_step=iter_no)

    def load_params(self, param_group, tag='iter', dir_name='saved', iter_no=None):
        weights_path = paths.get_saved_params_path(dir_name, param_group, tag, iter_no)
        self.weight_savers[param_group].restore(self.session, weights_path)
        return iter_no

    def log_custom_summary(self, logger_name, tag, simple_value, iter_no):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=tag, simple_value=simple_value),
        ])
        self.loggers[logger_name].add_summary(summary, global_step=iter_no)
