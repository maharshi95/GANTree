from __future__ import print_function, division
import torch as tr
import os, argparse, logging, json
from configs import Config, TrainConfig
from exp_context import ExperimentContext

print('mode:', 'gpu' if Config.use_gpu else 'cpu')

if Config.use_gpu:
    tr.set_default_tensor_type('torch.cuda.FloatTensor')

#### **Argument Parser**

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
print(json.dumps(args.__dict__, indent=4))

resume_flag = args.resume is not False
gpu_idx = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

#### **Set Experiment Context**

ExperimentContext.set_context(args.hyperparams, args.exp_name)
H = ExperimentContext.Hyperparams  # type: Hyperparams

#### **Set Logging**

logger = logging.getLogger(__name__)
LOG_FORMAT = "[{}: %(filename)s: %(lineno)3s] %(levelname)s: %(funcName)s(): %(message)s".format(ExperimentContext.exp_name)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

#### **Clear Logs and Results based on the argument flags**

from paths import Paths
from utils import bash_utils, model_utils

if 'all' in args.delete or 'logs' in args.delete or resume_flag is False:
    logger.warning('Deleting Logs...')
    bash_utils.delete_recursive(Paths.logs_base_dir)
    print('')

if 'all' in args.delete or 'results' in args.delete:
    logger.warning('Deleting all results in {}...'.format(Paths.results_base_dir))
    bash_utils.delete_recursive(Paths.results_base_dir)
    print('')

##### **Create required directories**

model_utils.setup_dirs()

##### **Model and Training related imports**

from dataloaders.factory import DataLoaderFactory
from base.hyperparams import Hyperparams

from models.toy.gt.gantree import GanTree
from models.toy.gt.named_tuples import DistParams
from models.toy.gan import ToyGAN

##### **Tensorboard Port**

ip = bash_utils.get_ip_address()
tboard_port = str(bash_utils.find_free_port(Config.base_port))
bash_utils.launchTensorBoard(Paths.logs_base_dir, tboard_port)
address = '{ip}:{port}'.format(ip=ip, port=tboard_port)
address_str = 'http://{}'.format(address)
tensorboard_msg = "Tensorboard active at http://%s:%s" % (ip, tboard_port)

##### **Dump Hyperparams file the experiments directory**

hyperparams_string_content = json.dumps(H.__dict__, default=lambda x: repr(x), indent=4, sort_keys=True)
# print(hyperparams_string_content)
with open(Paths.exp_hyperparams_file, "w") as fp:
    fp.write(hyperparams_string_content)

##### **Define Train Config**

train_config = TrainConfig(
    n_step_tboard_log=50,
    n_step_console_log=-1,
    n_step_validation=100,
    n_step_save_params=1000,
    n_step_visualize=500
)

##### **Create Gan Model and DataLoader for root GNode**

gan = ToyGAN.create_from_hyperparams('node0', H, '0')
dist_params = DistParams(gan.z_op_params[0], gan.z_op_params[1], 1.0, 1.0)
dl = DataLoaderFactory.get_dataloader(H.dataloader, H.input_size, H.z_size, H.batch_size, H.batch_size, supervised=True)
x_batch, _ = dl.random_batch('test', 2048)

##### **Create Gan Tree and GNode**G

tree = GanTree('gtree', ToyGAN, H, x_batch)
gnode = tree.create_child_node(dist_params, gan)

##### **Set Trainer for GNode**

gnode.set_trainer(dl, H, train_config)

print(tensorboard_msg)

gnode.train(30000)
