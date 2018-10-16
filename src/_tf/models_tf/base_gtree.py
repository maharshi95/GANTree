from __future__ import division, print_function
import os, logging
from _tf.models_tf import base_infergan

from _tf import paths
import tensorflow as tf

from abc import ABCMeta, abstractmethod
from tensorflow.contrib import framework as tf_framework

logger = logging.getLogger(__name__)


class BaseModel(base_infergan.BaseModel):
    __metaclass__ = ABCMeta

    def __init__(self, model_name, session=None, model_scope=None, shared_scope=""):
        self.model_name = model_name
        self.model_scope = model_name if model_scope is None else model_scope
        self.shared_scope = shared_scope
        # Weight Savers and Loaders
        self.param_groups = {}
        self.loggers = {}

        self.param_savers = {}
        self.weights_path = {}
        self.summary_nodes = {}
        self.session = session

    @property
    def name(self):
        return self.model_name

    @property
    def scope(self):
        return self.model_scope

    @property
    def private_scope(self):
        return self.model_scope + '/private'

    def __repr__(self):
        return ('Model[%s]' % self.model_name)

    @abstractmethod
    def build(self):
        return NotImplemented

    @abstractmethod
    def initiate_service(self, initialize_shared_variables=False):
        """
        Override this function in the subclass and make sure to call this method from the overridden method.
        Summary Writers, Weight Savers will all be initiatized in this function after the build() function has been called.
        :return:
        """
        if self.session is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            self.session = tf.Session(config=config)

        global_private_var_init_op = tf.variables_initializer(tf.global_variables(self.private_scope))
        global_shared_var_init_op = tf.variables_initializer(tf.global_variables(self.shared_scope))
        local_var_init_op = tf.variables_initializer(tf.local_variables(self.private_scope))

        var_init_ops = [global_private_var_init_op, local_var_init_op]

        if initialize_shared_variables:
            logger.info('Initializing shared variables at %s of %s' % (self.shared_scope, self))
            var_init_ops.append(global_shared_var_init_op)

        self.session.run(var_init_ops)

        uninit_var = self.session.run(tf.report_uninitialized_variables())

        if len(uninit_var) > 0:
            logger.warning('Found %d uninitialized variables: ' % len(uninit_var))
            for var in uninit_var:
                print(var)
            print('')

        for name in ['train', 'test']:
            self.add_logger(name, paths.log_writer_path(name, self.name))

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
        # type: (BaseModel) -> None
        logger.info('Loading Weights from %s to %s' % (parent_model.name, self.name))
        parent_model_param_values = parent_model.get_all_param_values(private_only=False, trainable_only=True)
        new_model_params = self.get_params(private_only=False, trainable_only=True)
        new_variable_names = map(lambda v: v.name[:-2], new_model_params)

        # logger.info('new variables in %s' % parent_model.name)
        # for var_name in new_variable_names:
        #     logger.info(var_name)

        variable_load_ops = []

        with tf.variable_scope("", reuse=True):
            for parent_var_name in parent_model_param_values:
                if 'private' in parent_var_name:  # Getting new name for private variable of model
                    new_variable_name = parent_var_name.replace(parent_model.model_name, self.model_name)
                else:  # Getting new name for shared variable of the model
                    tokens = parent_var_name.split('/')
                    shared_var_group_tag = tokens[tokens.index('shared') + 1]
                    new_variable_name = parent_var_name.replace(shared_var_group_tag, parent_model.model_name)
                if new_variable_name not in new_variable_names:
                    logger.warning('Oops!, {} not in var list'.format(new_variable_name))
                var_value = parent_model_param_values[parent_var_name]
                new_variable = tf.get_variable(new_variable_name)  # type: tf.Variable
                load_op = tf.assign(new_variable, var_value, name='Variable_Load_%s' % new_variable_name)
                variable_load_ops.append(load_op)
        self.session.run(variable_load_ops)
        return

    def load_params_from_checkpoints(self, dir_name='all', param_group='all', tag='iter', iter_no=None):
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
                    logger.warning('Oops!, {} not in var list'.format(new_variable_name))
                print(new_variable_name)
                if 'RMSProp' in new_variable_name:
                    print()
                    'ignoring...'
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

    def log_new_summary(self, logger_name, tag, simple_value, iter_no):
        if tag not in self.summary_nodes:
            self.summary_nodes[tag] = tf.summary.scalar(tag, tf.convert_to_tensor(simple_value))

        summary = self.session.run(self.summary_nodes[tag])

        self.loggers[logger_name].add_summary(summary, global_step=iter_no)

    def get_params(self, private_only, trainable_only):
        params = {
            True: {
                True: self.param_groups['private_trainable'],
                False: self.param_groups['private'],
            },
            False: {
                True: self.param_groups['trainable'],
                False: self.param_groups['all']
            }}[private_only][trainable_only]
        return params

    def get_all_param_values(self, private_only=True, trainable_only=True):
        """
        Fetches a python dict of type {parameter_name (str): parameter_value (float)}
        having parameter name as key and its current value as the value in dict.
        :param private_only: If true, only fetches the private variables of that model,
                              else fetches the shared (with some other model) params as well
        """
        param_values_dict = {}
        # scope = self.private_scope if private_only else self.model_scope
        # params = _tf.trainable_variables(scope) if trainable_only else _tf.global_variables(scope)
        params = self.get_params(private_only, trainable_only)
        values = self.session.run(params)
        for param, value in zip(params, values):
            param_values_dict[param.name[:-2]] = value
        return param_values_dict

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return self.session.run(fetches, feed_dict, options, run_metadata)
