from enum import Enum

from . import base


class HyperparamsFactory:
    __dict = {
        'bcgan': base.Hyperparams(),
    }

    class_type = base.Hyperparams

    @classmethod
    def get_hyperparams(cls, name):
        # type: (str) -> base.Hyperparams
        return cls.__dict[name]
