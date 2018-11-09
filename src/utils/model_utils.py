from __future__ import print_function, absolute_import
from paths import Paths as paths
from . import bash_utils


def setup_dirs(log_flag=True):
    bash_utils.create_dir(paths.results_base_dir, log_flag)
    bash_utils.create_dir(paths.logs_base_dir, log_flag)
    bash_utils.create_dir(paths.saved_weights_dir, log_flag)
    bash_utils.create_dir(paths.all_weights_dir, log_flag)
    bash_utils.create_dir(paths.results_base_dir, log_flag)
    bash_utils.create_dir(paths.temp_dir, log_flag)
