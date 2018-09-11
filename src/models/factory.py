from __future__ import print_function
import os, sys, importlib


class ModelFactory:

    @classmethod
    def get_model(cls, name):
        # type: (str) -> Type[BCGanModel]
        module_name = __package__ + '.' + name
        print('importing model %s' % module_name)
        module = importlib.import_module(module_name)
        Model = module.Model
        return Model
