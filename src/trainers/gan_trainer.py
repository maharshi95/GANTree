from __future__ import print_function, division
import torch as tr

from base.model import BaseGan
from base.trainer import BaseTrainer
from base.dataloader import BaseDataLoader


class GanTrainer(BaseTrainer):
    def __init__(self, model, data_loader, n_iterations):
        # type: (BaseGan, BaseDataLoader, int) -> None
        super(GanTrainer, self).__init__(model, data_loader, n_iterations)

    def train(self):
        dl = self.data_loader
        model = self.model

        for iter_no in range(self.n_iterations):
            x_train = tr.Tensor(dl.next_batch('train'))
            z_train = tr.Tensor(dl.next_batch('train'))

            if iter_no % 20 < 15:
                c_loss = model.step_train_autoencoder(x_train, z_train)
                g_loss = model.step_train_generator(z_train)
            else:
                # pass
                d_loss = model.step_train_discriminator(x_train, z_train)
            g_acc, d_acc = model.get_accuracies(x_train, z_train)

            x_test = dl.next_batch('test')

            if iter_no % 100 == 99:
                print('Step', iter_no + 1)
                print('Gen  Accuracy:', g_acc.item())
                print('Disc Accuracy:', d_acc.item())
