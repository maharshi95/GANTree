import os, argparse, logging, json

parser = argparse.ArgumentParser()

parser.add_argument('-g', '--gpu', default=0, help='index of the gpu to be used. default: 0')
parser.add_argument('-r', '--resume', nargs='?', const=True, default=False,
                    help='if present, the training resumes from the latest step, '
                         'for custom step number, provide it as argument value')
parser.add_argument('-d', '--delete', nargs='+', default=[], choices=['logs', 'weights', 'results', 'all'],
                    help='delete the entities')
parser.add_argument('-w', '--weights', nargs='?', default='iter', choices=['iter', 'best_gen', 'best_pred'],
                    help='weight type to load if resume flag is provided. default: iter')
parser.add_argument('-hp', '--hyperparams', required=True, help='hyperparam class to use from HyperparamFactory')
parser.add_argument('-en', '--exp_name', default=None, help='experiment name. if not provided, it is taken from Hyperparams')

args = parser.parse_args()

resume_flag = args.resume is not False

from exp_context import ExperimentContext

ExperimentContext.set_context(args.hyperparams, args.exp_name)
H = ExperimentContext.Hyperparams  # type: Hyperparams

logger = logging.getLogger(__name__)
LOG_FORMAT = "[{}: %(filename)s: %(lineno)3s] %(levelname)s: %(funcName)s(): %(message)s".format(ExperimentContext.exp_name)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

gpu_idx = str(args.gpu)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

import paths
from utils import bash_utils, model_utils

if 'all' in args.delete or 'logs' in args.delete or resume_flag is False:
    logger.warning('Deleting Logs...')
    bash_utils.delete_recursive(paths.logs_base_dir)
    print('')

if 'all' in args.delete or 'results' in args.delete:
    logger.warning('Deleting all results in {}...'.format(paths.results_base_dir))
    bash_utils.delete_recursive(paths.results_base_dir)
    print('')

model_utils.setup_dirs()

from dataloaders.factory import DataLoaderFactory
from base.hyperparams import Hyperparams

from models.toy.nets import ToyGAN
from trainers.gan_trainer import GanTrainer
from trainers.gan_trainer import TrainConfig

train_config = TrainConfig(
    n_step_tboard_log=100,
    n_step_console_log=100,
    n_step_validation=200,
    n_step_save_params=200,
    n_step_visualize=200
)

gan = ToyGAN('gan')
dl = DataLoaderFactory.get_dataloader(H.dataloader, H.input_size, H.z_size, H.batch_size, H.batch_size)
trainer = GanTrainer(data_loader=dl, model=gan, hyperparams=H, train_config=train_config)

hyperparams_string_content = json.dumps(H.__dict__, default=lambda x: repr(x), indent=4, sort_keys=True)
print(hyperparams_string_content)
with open(paths.exp_hyperparams_file, "w") as fp:
    fp.write(hyperparams_string_content)

trainer.train()
