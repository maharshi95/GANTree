import importlib
from . import base



class HyperparamsFactory:

    class_type = base.Hyperparams

    @classmethod
    def get_hyperparams(cls, name):
        # type: (str) -> bcgan_2d.Hyperparams
        # return cls.__dict[name]

        module_name = __package__ + '.' + name
        print('importing hyperparams %s' % module_name)
        module = importlib.import_module(module_name)
        Hyperparams = module.Hyperparams
        return Hyperparams
