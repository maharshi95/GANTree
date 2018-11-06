import torch as tr
import inspect

from configs import Config


def get_numpy(val):
    if isinstance(val, list) or isinstance(val, tuple):
        return map(get_numpy, val)
    if isinstance(val, tr.Tensor):
        return val.cpu().detach().numpy() if tr.cuda.is_available() else val.detach().numpy()
    return val


def make_tensor(use_gpu=Config.use_gpu):
    """
    A 2nd order decorator creator, for managing following things:
        - Adds an extra argument ``numpy=True`` to input function, if set False, returns torch.Tensor, else numpy.ndarray
        - If input tensor not an instance of torch.Tensor, converts it to torch.Tensor before passing to the model
        - If input tensor is empty, returns an empty Tensor/Array
    :return: A function decorator with described properties
    >>> @make_tensor(use_gpu=True)
        def encode(self, x):
            return self.encoder(x)
    """

    def inner_decorator(orig_func):
        def inner_func(self, input_tensor, numpy=True, *args, **kwargs):
            if not isinstance(input_tensor, tr.Tensor):
                input_tensor = tr.Tensor(input_tensor)
                if use_gpu:
                    input_tensor = input_tensor.cuda()

            if len(input_tensor) > 0:
                ret = orig_func(self, input_tensor, *args, **kwargs)
            else:
                ret = tr.Tensor([])

            if numpy:
                ret = map(get_numpy, ret) if isinstance(ret, tuple) else get_numpy(ret)
            return ret

        return inner_func

    return inner_decorator


def tensorify(use_gpu=Config.use_gpu):
    def inner_decorator(orig_func):
        def inner_func(self, *args, **kwargs):
            args = map(lambda arg: tr.tensor(arg, dtype=tr.float32), args)
            if use_gpu:
                args = map(lambda arg: arg.cuda(), args)

            ret = orig_func(self, *args, **kwargs)
            ret = map(get_numpy, ret) if isinstance(ret, tuple) else get_numpy(ret)
            return ret

        return inner_func

    return inner_decorator


def tensor_output(use_gpu=True):
    def outer(f):
        def inner(*args, **kwargs):
            ret = f(*args, **kwargs)
            if isinstance(ret, tuple):
                ret = map(lambda v: v.cuda() if use_gpu else v, map(tr.Tensor, ret))
            else:
                ret = tr.Tensor(ret)
                if use_gpu:
                    ret = ret.cuda()
            return ret

        return inner

    return outer


def numpy_output(f):
    def inner_func(*args, **kwargs):
        ret = f(*args, **kwargs)

        if isinstance(ret, tuple) or isinstance(ret, list):
            ret = map(get_numpy, ret)
        else:
            ret = get_numpy(ret)
        return ret

    return inner_func


class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)
