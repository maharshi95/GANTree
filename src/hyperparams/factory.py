import importlib
import base.hyperparams as base


class HyperparamsFactory:
    class_type = base.Hyperparams

    @classmethod
    def get_hyperparams(cls, module_name):
        # type: (str) -> bcgan_2d.Hyperparams
        # return cls.__dict[name]

        print('importing hyperparams %s' % module_name)
        module = importlib.import_module(module_name)
        Hyperparams = module.Hyperparams
        return Hyperparams
