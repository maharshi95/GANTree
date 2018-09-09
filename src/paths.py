import os.path as osp

# Make sure to put the corresponding line in function src.utils.model_utils.setup_dirs after adding/modifying a directory path

# data_base_dir = '../data'
data_base_dir = '../../datasets'
logs_base_dir = '../logs/'
weights_base_dir = '../weights/'
results_base_dir = '../results/'

temp_dir = osp.join(results_base_dir, '../temp/')

# NOT SET YET
train_data = ''
test_data = ''


def log_writer_path(writer_name):
    return osp.join(logs_base_dir, writer_name)


all_weights_dir = osp.join(weights_base_dir, 'all/')
saved_weights_dir = osp.join(weights_base_dir, 'saved/')

encoder_best_weights = osp.join(all_weights_dir, 'encoder_best')

encoder_iter_weights = osp.join(all_weights_dir, 'encoder_iter')

weights_dir_paths = {
    'all': all_weights_dir,
    'saved': saved_weights_dir,
}


def get_result_path(path):
    return osp.join(results_base_dir, path)


def get_recon_path(path):
    return osp.join(recon_dir, path)


def get_hist_path(path):
    return osp.join(hist_dir, path)


def get_gen_path(path):
    return osp.join(gen_dir, path)


def get_temp_file_path(path):
    return osp.join(temp_dir, path)


def get_nn_path(path):
    return osp.join(nn_dir, path)


def get_saved_params_path(dir_name, net_name, weight_label, iter_no):
    dir_path = weights_dir_paths[dir_name]
    if iter_no is None:
        return osp.join(dir_path, '%s_%s' % (net_name, weight_label))
    else:
        return osp.join(dir_path, '%s_%s-%d' % (net_name, weight_label, iter_no))
