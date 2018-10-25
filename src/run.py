import os, argparse, logging, json
import torch as tr
from configs import Config

if Config.use_gpu:
    print('mode: GPU')
    tr.set_default_tensor_type('torch.cuda.FloatTensor')

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
    n_step_tboard_log=50,
    n_step_console_log=500,
    n_step_validation=100,
    n_step_save_params=1000,
    n_step_visualize=2000
)
cor = -0.6
z_op_params = tr.zeros(H.z_size), tr.Tensor([[1.0, cor],
                                             [cor, 1.0]])
gan = ToyGAN('gan', z_op_params, z_bounds=H.z_bounds)
dl = DataLoaderFactory.get_dataloader(H.dataloader, H.input_size, H.z_size, H.batch_size, H.batch_size, supervised=True)
trainer = GanTrainer(data_loader=dl, model=gan, hyperparams=H, train_config=train_config)

# Dump Hyperparams file the experiments directory
hyperparams_string_content = json.dumps(H.__dict__, default=lambda x: repr(x), indent=4, sort_keys=True)
print(hyperparams_string_content)
with open(paths.exp_hyperparams_file, "w") as fp:
    fp.write(hyperparams_string_content)


def launchTensorBoard(path, port):
    import os
    os.system('tensorboard --logdir {} --port {}'.format(path, port))
    return


import threading

t = threading.Thread(target=launchTensorBoard, args=([paths.logs_base_dir, 7000]))
t.start()

trainer.train()
