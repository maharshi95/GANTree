from broken_segments import BrokenSegmentsDataLoader
from broken_circle import BrokenCircleDataLoader
from dataloaders.base import BaseDataLoader
from multi_normal import TwoGaussiansDataLoader


class DataLoaderFactory(object):
    __dict = {
        'broken_circle': BrokenCircleDataLoader,
        'broken_segments': BrokenSegmentsDataLoader,
        'two_gaussian': TwoGaussiansDataLoader,
    }

    @classmethod
    def get_dataloader(cls, name, input_size=1, latent_size=1):
        # type: (str, int, int) -> BaseDataLoader
        DL = cls.__dict[name]
        return DL(input_size, latent_size)
