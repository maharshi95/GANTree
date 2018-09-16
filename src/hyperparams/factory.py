import importlib
from . import bcgan_2d


class HyperparamsFactory:
    __dict = {
        'bcgan': bcgan_2d.Hyperparams(),
    }

    class_type = bcgan_2d.Hyperparams

    @classmethod
    def get_hyperparams(cls, name):
        # type: (str) -> bcgan_2d.Hyperparams
        # return cls.__dict[name]

        module_name = __package__ + '.' + name
        print('importing hyperparams %s' % module_name)
        module = importlib.import_module(module_name)
        Hyperparams = module.Hyperparams
        return Hyperparams
