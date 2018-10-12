from types import ModuleType
from hyperparams.factory import HyperparamsFactory
from models.factory import ModelFactory


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
    def set_context(cls, hyperparams_name, exp_name=None):
        if isinstance(hyperparams_name, ModuleType):
            try:
                cls.hyperparams_name = hyperparams_name.__name__
                cls.Hyperparams = hyperparams_name.Hyperparams
            except Exception as ex:
                print('module has no attribute Hyperparams: %s' % hyperparams_name.__name__)
                raise ex
        else:
            if hyperparams_name.endswith('.py'):
                hyperparams_name = '.'.join(hyperparams_name.split('/'))[:-3]
            cls.hyperparams_name = hyperparams_name
            cls.Hyperparams = HyperparamsFactory.get_hyperparams(cls.hyperparams_name)
        cls.exp_name = exp_name or cls.Hyperparams.exp_name
        cls.Model = ModelFactory.get_model(cls.Hyperparams.model)

    @classmethod
    def __repr__(cls):
        return "ExperimentContext: hyperparams_name: {}, exp_name: {}".format(cls.hyperparams_name, cls.exp_name)

    @classmethod
    def get_hyperparams(cls):
        return cls.Hyperparams

    @classmethod
    def get_model_class(cls):
        return cls.Model

    tp = type(ModelFactory)
