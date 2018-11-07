from base.dataloader import BaseDataLoader


class CelebA(BaseDataLoader):

    def __init__(self, input_size=1, latent_size=2, train_batch_size=32, test_batch_size=32, get_data=None, *args, **kwargs):
        self.get_data = get_data
        super(CelebA, self).__init__(input_size, latent_size, train_batch_size, test_batch_size, *args,
                                     **kwargs)

    # TODO  getdata
