from __future__ import division, print_function, absolute_import
import os, time, argparse, logging, traceback, json

import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from exp_context import ExperimentContext

# Setting up Argument parser

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

ExperimentContext.set_context(args.hyperparams, args.exp_name)

logger = logging.getLogger(__name__)
LOG_FORMAT = "[{}: %(filename)s: %(lineno)3s] %(levelname)s: %(funcName)s(): %(message)s".format(ExperimentContext.exp_name)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

import paths
from utils import viz_utils
from utils import bash_utils, model_utils
from dataloaders.factory import DataLoaderFactory

gpu_idx = str(args.gpu)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

resume_flag = args.resume is not False

if 'all' in args.delete or 'logs' in args.delete or resume_flag is False:
    logger.warning('Deleting Logs...')
    bash_utils.delete_recursive(paths.logs_base_dir)
    print('')

if 'all' in args.delete or 'results' in args.delete:
    logger.warning('Deleting all results in {}...'.format(paths.results_base_dir))
    bash_utils.delete_recursive(paths.results_base_dir)
    print('')

model_utils.setup_dirs()

Model = ExperimentContext.get_model_class()
H = ExperimentContext.get_hyperparams()
print('input_size:', H.input_size, 'latent_size:', H.z_size)
dl = DataLoaderFactory.get_image_dataloader(H.dataloader, H.batch_size_train, H.batch_size_test)

# Writing the Hyperparams file to experiment directory
hyperparams_string_content = json.dumps(H.__dict__, default=lambda x: repr(x), indent=4, sort_keys=True)
print(hyperparams_string_content)
with open(paths.exp_hyperparams_file, "w") as fp:
    fp.write(hyperparams_string_content)

# Building the Model and initiating Model Service
model = Model('growing-gans')
model.build()
model.initiate_service()

print('Model service initiated...')

# Resume the training from a specific iteration - Loading the trained weights
if resume_flag is False:
    iter_no = 0
else:
    try:
        iter_no = None if args.resume is True else args.resume
        iter_no = model.load_params_from_history(dir_name='all', param_group='all', iter_no=iter_no)
        logger.info('Loading network weights from all-%d' % iter_no)
    except Exception as ex:
        traceback.print_exc()
        logger.error(ex)
        logger.error('Some Problem Occured while resuming... starting from iteration 1')
        raise Exception('Not found...')

if 'all' in args.delete or 'weights' in args.delete:
    logger.warning('Deleting all weights in {}...'.format(paths.all_weights_dir))
    bash_utils.delete_recursive(paths.all_weights_dir)
    model_utils.setup_dirs()
    print('')

print('Train Test Data loaded...')

max_epochs = 100000

n_step_console_log = 20
n_step_tboard_log = 10
n_step_validation = 50
n_step_iter_save = 2000
n_step_visualize = 100
# n_step_generator_decay = 1500

en_loss_history = []
de_loss_history = []
di_loss_history = []
d_acc_history = []
g_acc_history = []
gen_loss_history = []

z_seed = {
    'train': dl.get_z_dist(3, dim=H.z_size, dist_type=H.z_dist_type),
    'test': dl.get_z_dist(3, dim=H.z_size, dist_type=H.z_dist_type)
}
x_seed = {
    'train': dl.random_batch(split='train', batch_size=3),
    'test': dl.random_batch(split='test', batch_size=3)
}
# x_seed['test'] = x_seed['train']

BN_train = True
BN_test = False
print('BN_train: ',BN_train)
print('BN_test: ',BN_test)

while iter_no < max_epochs:
    iter_no += 1

    iter_time_start = time.time()
    # print('x_train',x_train.shape)
    # print('x_test',x_test.shape)
    x_train, x_test = dl.get_data()
    # x_test = x_train

    z_train = dl.get_z_dist(x_train.shape[0], dim=H.z_size, dist_type=H.z_dist_type)
    z_test = dl.get_z_dist(x_test.shape[0], dim=H.z_size, dist_type=H.z_dist_type)

    # z_test = z_train

    # print('z_train', z_train.shape)
    # print('z_test', z_test.shape)

    train_inputs = x_train, z_train, BN_train
    test_inputs = x_test, z_test, BN_test

    if (iter_no % H.combined_iter_count) < H.gen_iter_count:
        model.step_train_encoder(train_inputs)
        model.step_train_decoder(train_inputs)
    else:
        model.step_train_discriminator(train_inputs)

    # if (iter_no % n_step_generator_decay) == 0:
    #     n_step_generator = max(n_step_generator - 1, 1)

    train_losses = model.compute_losses(train_inputs, model.network_loss_variables)
    #
    if iter_no % n_step_console_log == 0:
        print('%s: Step %i: Encoder Loss: %f' % (ExperimentContext.exp_name, iter_no, train_losses[0]))
        print('%s: Step %i: Disc Acc: %f' % (ExperimentContext.exp_name, iter_no, train_losses[1]))
        print('%s: Step %i: Gen  Acc: %f' % (ExperimentContext.exp_name, iter_no, train_losses[2]))
        print('%s: Step %i: x_recon Loss: %f' % (ExperimentContext.exp_name, iter_no, train_losses[3]))
        print('%s: Step %i: z_recon Loss: %f' % (ExperimentContext.exp_name, iter_no, train_losses[4]))
        print()

    if iter_no % n_step_tboard_log == 0:
        model.get_logger('train').add_summary(train_losses[-1], iter_no)

    if iter_no % n_step_visualize == 0:
        for split in ['train', 'test']:
            x_gt = x_seed[split]
            bn_flag = BN_train if split == 'train' else BN_test
            x_recon_seed = model.reconstruct_x(x_seed[split], flag=bn_flag)
            print('loss', np.mean((x_recon_seed - x_seed[split]) ** 2))
            x_gen_seed = model.decode(z_seed[split], flag=bn_flag)
            # x_recon_seed = model.reconstruct_x(x_seed[split])
            # x_gen_seed = model.decode(z_seed[split])
            model.log_images(split, [x_gt, x_recon_seed, x_gen_seed], iter_no)

    if iter_no % n_step_validation == 0:
        test_losses = model.compute_losses(test_inputs, model.network_loss_variables)
        model.get_logger('test').add_summary(test_losses[-1], iter_no)

    if iter_no % n_step_iter_save == 0:
        model.save_params(iter_no=iter_no)

    iter_time_end = time.time()

    if iter_no % n_step_console_log == 0:
        print('%s: Single Iter Time: %.4f' % (ExperimentContext.exp_name, iter_time_end - iter_time_start))
