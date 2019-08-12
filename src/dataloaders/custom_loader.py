from base.dataloader import BaseDataLoader

class CustomDataLoader(BaseDataLoader):

    @staticmethod
    def create_from_parent(dataloader, data):
        # type: (BaseDataLoader, tuple) -> CustomDataLoader
        """
        Creates a new ``CustomDataloader`` object by retaining attributes like
        ``input_size``, ``latent_size``, ``batch_size``, ``supervised`` from the input ``dataloader`` and populating the new ``data``

        :param data: A tuple of 4 objects if supervised - train_data, test_data, train_labels, test_labels
                        else a tuple of 2 objects - ``train_data, test_data``

        """
        supervised = len(data) == 4

        custom_dataloader = CustomDataLoader(img_size=dataloader.img_size,
                                             latent_size=dataloader.latent_size,
                                             train_batch_size=dataloader.batch_size['train'],
                                             test_batch_size=dataloader.batch_size['test'],
                                             supervised=supervised,
                                             get_data=lambda: data)

        return custom_dataloader

    def __init__(self, img_size=1, latent_size=2, train_batch_size=64, test_batch_size=64, get_data=None, *args, **kwargs):
        self.get_data = get_data
        
        super(CustomDataLoader, self).__init__(img_size, latent_size, train_batch_size, test_batch_size, *args,
                                               **kwargs)
        
        self.__custom_data = {
            'train': None,
            'test': None
        }
        
        self.__custom_labels = {
            'train': None,
            'test': None
        }
    

    def populate_data(self, split, x_data, labels):
        self.__custom_data[split] = x_data
        self.__custom_labels[split] = labels
        all_data = self.get_data()
        self.update_data(*all_data)

    def get_data(self, train_ratio=0.6):
        return (
            self.__custom_data['train'],
            self.__custom_data['test'],
            self.__custom_labels['train'],
            self.__custom_labels['test'],
        )
