import inspect
from hyperparams.factory import HyperparamsFactory


class ExperimentContext:
    """
    A common unique class across an entire experiment process.
    This Static class holds the hyper-parameters, model and experiment name information for the run.
    A Model class implementation must use the hyperparams via this Static class.
    """
    hyperparams_name = None
    Hyperparams = None  # type: Type[HyperparamsFactory.class_type]
    Model = None
    exp_name = None

    @classmethod
    def set_context(cls, hyperparams, exp_name=None):
        if inspect.isclass(hyperparams):
            cls.Hyperparams = hyperparams
            cls.hyperparams_name = 'dynamic'
            print 'loaded HP from class'

        elif inspect.ismodule(hyperparams):
            print 'loading HP from module'
            try:
                cls.hyperparams_name = hyperparams.__name__
                cls.Hyperparams = hyperparams.Hyperparams
            except Exception as ex:
                print('module has no attribute Hyperparams: %s' % hyperparams.__name__)
                raise ex
        else:
            print 'loading HP from file'
            if hyperparams.endswith('.py'):
                hyperparams = '.'.join(hyperparams.split('/'))[:-3]
            cls.hyperparams_name = hyperparams
            cls.Hyperparams = HyperparamsFactory.get_hyperparams(cls.hyperparams_name)
        cls.exp_name = exp_name or cls.Hyperparams.exp_name

    @classmethod
    def __repr__(cls):
        return "ExperimentContext: hyperparams_name: {}, exp_name: {}".format(cls.hyperparams_name, cls.exp_name)

    @classmethod
    def get_hyperparams(cls):
        return cls.Hyperparams

    @classmethod
    def get_model_class(cls):
        return cls.Model
