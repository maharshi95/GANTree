from broken_segments import BrokenSegmentsDataLoader
from broken_circle import BrokenCircleDataLoader
from dataloaders.base import BaseDataLoader
from multi_normal import TwoGaussiansDataLoader, FourGaussiansDataLoader, FourSymGaussiansDataLoader
from dataloader_mnist import MNISTDataLoader


class DataLoaderFactory(object):
    __dict = {
        'broken_circle': BrokenCircleDataLoader,
        'broken_segments': BrokenSegmentsDataLoader,
        'two_gaussian': TwoGaussiansDataLoader,
        'four_gaussian': FourGaussiansDataLoader,
        'four_gaussian_sym': FourSymGaussiansDataLoader,
        'mnist': MNISTDataLoader
    }

    @classmethod
    def get_toy_dataloader(cls, name, input_size=1, latent_size=1):
        # type: (str, int, int) -> (BaseDataLoader | MNISTDataLoader)
        DL = cls.__dict[name]
        if name == 'mnist':
            return DL()
        return DL(input_size, latent_size)

    @classmethod
    def get_image_dataloader(cls, name, *args):
        # type: (str, object) -> MNISTDataLoader
        DL = cls.__dict[name]
        return DL(*args)
