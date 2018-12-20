from broken_segments import BrokenSegmentsDataLoader
from broken_circle import BrokenCircleDataLoader
from base.dataloader import BaseDataLoader
from dataloaders.colored import FaceBedDataLoader
from dataloaders.mnist import MixedMnistDataLoader
from .multi_normal import TwoGaussiansDataLoader, FourGaussiansDataLoader, FourSymGaussiansDataLoader, NineGaussiansDataLoader
from .celeba import CelebA
from .mnist import MnistDataLoader, FashionMnistDataLoader


class DataLoaderFactory(object):
    __dict = {
        # Toy Datasets
        'broken_circle': BrokenCircleDataLoader,
        'broken_segments': BrokenSegmentsDataLoader,
        'two_gaussian': TwoGaussiansDataLoader,
        'four_gaussian': FourGaussiansDataLoader,
        'four_gaussian_sym': FourSymGaussiansDataLoader,
        'nine_gaussian': NineGaussiansDataLoader,

        # Single Channel Image Datasets
        'mnist': MnistDataLoader,
        'fashion': FashionMnistDataLoader,
        'mixed_mnist': MixedMnistDataLoader,
        'facebed': FaceBedDataLoader,

        # RGB Channel Image Datasets
        'celeba': CelebA,
    }

    @classmethod
    def get_dataloader(cls, name, input_size=1, latent_size=1, *args, **kwargs):
        # type: (str, int, int, *tuple, **dict) -> BaseDataLoader
        DL = cls.__dict[name]
        return DL(input_size=input_size, latent_size=latent_size, *args, **kwargs)

    @classmethod
    def get_img_dataloader(cls, name, *args, **kwargs):
        DL = cls.__dict[name]
        return DL(*args, **kwargs)
