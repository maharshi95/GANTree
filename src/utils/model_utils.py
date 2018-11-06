from paths import Paths as paths
from . import bash_utils


def setup_dirs():
    bash_utils.create_dir(paths.results_base_dir)
    bash_utils.create_dir(paths.logs_base_dir)
    bash_utils.create_dir(paths.saved_weights_dir)
    bash_utils.create_dir(paths.all_weights_dir)
    bash_utils.create_dir(paths.results_base_dir)
    bash_utils.create_dir(paths.temp_dir)
    print()
