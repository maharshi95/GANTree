from __future__ import division, print_function, absolute_import
import os, argparse, logging
import traceback

import numpy as np
from utils import bash_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-g', '--gpu', default=0, help='index of the gpu to be used. default: 0')
parser.add_argument('-r', '--resume', nargs='?', const=True, default=False,
                    help='if present, the training resumes from the latest step, '
                         'for custom step number, provide it as argument value')
parser.add_argument('-d', '--delete', nargs='+', default=[], choices=['logs', 'weights'], help='delete the entities')
parser.add_argument('-w', '--weights', nargs='?', default='iter', choices=['iter', 'best_gen', 'best_pred'],
                    help='weight type to load if resume flag is provided. default: iter')

args = parser.parse_args()

gpu_idx = str(args.gpu)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

resume_flag = args.resume is not False

if 'logs' in args.delete or resume_flag is False:
    logger.warning('Deleting Logs...')
    bash_utils.delete_recursive(paths.logs_base_dir)
    print('')

if 'weights' in args.delete:
    logger.warning('Deleting all weights in {}...'.format(paths.all_weights_dir))
    bash_utils.delete_recursive(paths.all_weights_dir)
    print('')

from data import DataLoader
from models.bcgan import Model

dl = DataLoader()

model = Model('growing-gans')
model.build()
model.initiate_service()
print('Model service initiated...')

if resume_flag is not False:
    try:
        if args.resume is True:
            iter_no = model.load_params(dir_name='all', param_group='all')
        else:
            iter_no = int(args.resume)
            iter_no = model.load_params(dir_name='all', param_group='all', iter_no=iter_no)

        logger.info('Loading network weights from all-%d' % iter_no)
    except Exception as ex:
        traceback.print_exc()
        logger.error(ex)
        logger.error('Some Problem Occured while resuming... starting from iteration 1')
        raise Exception('Not found...')

x_train, x_test = dl.broken_circle()
print('Train Test Data loaded...')

epochs = 200

n_step_console_log = 10
n_step_tboard_log = 1
n_step_validation = 1
n_step_iter_save = 20

en_loss_history = []
de_loss_history = []
di_loss_history = []
d_acc_history = []
g_acc_history = []
gen_loss_history = []

for iter_no in range(epochs):
    iter_no += 1

    z_train = np.random.uniform(-1, 1, [x_train.shape[0], 1])
    z_test = np.random.uniform(-1, 1, [x_test.shape[0], 1])

    train_inputs = x_train, z_train
    test_inputs = x_test, z_test

    model.step_train_autoencoder(train_inputs)

    if (iter_no % 10) == 0:
        model.step_train_adv_generator(train_inputs)
    else:
        model.step_train_discriminator(train_inputs)

    network_losses = [
        model.encoder_loss,
        model.disc_acc,
        model.gen_acc,
        model.x_recon_loss,
        model.z_recon_loss,
        model.summaries
    ]

    train_losses = model.compute_losses(train_inputs, network_losses)
    test_losses = model.compute_losses(test_inputs, network_losses)

    if iter_no % n_step_console_log == 0:
        print('Step %i: Encoder Loss: %f' % (iter_no, train_losses[0]))
        print('Step %i: Disc Acc: %f' % (iter_no, train_losses[1]))
        print('Step %i: Gen  Acc: %f' % (iter_no, train_losses[2]))
        print('Step %i: x_recon Loss: %f' % (iter_no, train_losses[3]))
        print('Step %i: z_recon Loss: %f' % (iter_no, train_losses[4]))
        print()

    if iter_no % n_step_tboard_log == 0:
        model.get_logger('train').add_summary(train_losses[-1], iter_no)
        model.get_logger('test').add_summary(test_losses[-1], iter_no)

    if iter_no % n_step_iter_save:
        model.save_params(iter_no=iter_no)
