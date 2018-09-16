from enum import Enum

from . import base
from . import base_dc1


class HyperparamsFactory:  # change
    __dict = {
        'bcgan': base.Hyperparams(),
        'dcgan': base_dc1.Hyperparams()
    }

    class_type = base.Hyperparams

    @classmethod
    def get_hyperparams(cls, name):
        # type: (str) -> base.Hyperparams
        return cls.__dict[name]
