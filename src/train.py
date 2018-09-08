import os
import numpy as np
import tensorflow as tf

from data import DataLoader
from models.bcgan import Model

del_logs = True

if del_logs:
    os.system('rm -r logs/*')

dl = DataLoader()

model = Model('growing-gans')
model.build()
model.initiate_service()

x_train, x_test = dl.broken_circle()
epochs = 20000
n_step_display = 1000
n_step_validation = 1

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
        model.decoder_loss,
        model.disc_acc,
        model.gen_acc,
        model.x_recon_loss,
        model.z_recon_loss,
        model.summaries
    ]

    train_losses = model.compute_losses(train_inputs, network_losses)
    test_losses = model.compute_losses(test_inputs, network_losses)

    if iter_no % n_step_display == 0:
        print('Step %i: Encoder Loss: %f' % (iter_no, train_losses[0]))
        print('Step %i: Disc Acc: %f' % (iter_no, train_losses[1]))
        print('Step %i: Gen  Acc: %f' % (iter_no, train_losses[2]))
        print('Step %i: Disc Loss: %f' % (iter_no, train_losses[3]))
        print('Step %i: Gen  Loss: %f' % (iter_no, train_losses[4]))
