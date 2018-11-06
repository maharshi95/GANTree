from broken_segments import BrokenSegmentsDataLoader
from broken_circle import BrokenCircleDataLoader
from base.dataloader import BaseDataLoader
from .multi_normal import TwoGaussiansDataLoader, FourGaussiansDataLoader, FourSymGaussiansDataLoader, NineGaussiansDataLoader
from .dataloader_mnist import MNISTDataLoader


class DataLoaderFactory(object):
    __dict = {
        'broken_circle': BrokenCircleDataLoader,
        'broken_segments': BrokenSegmentsDataLoader,
        'two_gaussian': TwoGaussiansDataLoader,
        'four_gaussian': FourGaussiansDataLoader,
        'four_gaussian_sym': FourSymGaussiansDataLoader,
        'nine_gaussian': NineGaussiansDataLoader,
        'mnist': MNISTDataLoader
    }

    @classmethod
    def get_dataloader(cls, name, input_size=1, latent_size=1, *args, **kwargs):
        # type: (str, int, int, *tuple, **dict) -> BaseDataLoader
        DL = cls.__dict[name]
        if name == 'mnist':
            return DL()
        return DL(input_size, latent_size, *args, **kwargs)
