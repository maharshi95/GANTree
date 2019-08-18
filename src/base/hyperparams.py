import torch as tr


class Hyperparams:
    """
    Base Hyperparams class.
    It uses base version of bcgan with 1D x space and z space
    """
    dtype = float

    @classmethod
    def z_means(cls):
        return tr.zeros(cls.z_size)

    @classmethod
    def z_cov(cls, sign='0'):
        cov = tr.eye(cls.z_size)
        cor = {
            '+': cls.cor,
            '-': -cls.cor,
            '0': 0.0
        }[sign]
        cov[0, 1] = cov[1, 0] = cor
        return cov
