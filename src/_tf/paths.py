import os.path as osp
from exp_context import ExperimentContext

exp_name = ExperimentContext.exp_name

# Make sure to put the corresponding line in function src.utils.model_utils.setup_dirs after adding/modifying a directory path

# data_base_dir = '../data'
data_base_dir = '../../datasets'
experiments_base_dir = '../experiments'
current_exp_dir = osp.join(experiments_base_dir, exp_name)

logs_base_dir = osp.join(experiments_base_dir, exp_name, 'logs')
weights_base_dir = osp.join(experiments_base_dir, exp_name, 'weights')
results_base_dir = osp.join(experiments_base_dir, exp_name, 'results')
exp_hyperparams_file = osp.join(experiments_base_dir, exp_name, 'hyperparams.json')

temp_dir = osp.join(results_base_dir, '.temp')

# NOT SET YET
train_data = ''
test_data = ''


def log_writer_path(writer_name, model_name=None):
    model_name = model_name or ''
    return osp.join(logs_base_dir, model_name, writer_name)


all_weights_dir = osp.join(weights_base_dir, 'all/')
saved_weights_dir = osp.join(weights_base_dir, 'saved/')

encoder_best_weights = osp.join(all_weights_dir, 'encoder_best')

encoder_iter_weights = osp.join(all_weights_dir, 'encoder_iter')

weights_dir_paths = {
    'all': all_weights_dir,
    'saved': saved_weights_dir,
}


def get_result_path(path, model_name=None):
    model_name = model_name or ''
    return osp.join(results_base_dir, model_name, path)


def get_temp_file_path(path, model_name=None):
    model_name = model_name or ''
    return osp.join(temp_dir, model_name, path)


def get_saved_params_path(dir_name, model_name, net_name, weight_label, iter_no):
    dir_path = weights_dir_paths[dir_name]
    if iter_no is None:
        return osp.join(dir_path, model_name, '%s_%s' % (net_name, weight_label))
    else:
        return osp.join(dir_path, model_name, '%s_%s-%d' % (net_name, weight_label, iter_no))