from hyperparams.factory import HyperparamsFactory
from models.factory import ModelFactory


class ExperimentContext:
    """
    A common unique class across an entire experiment process.
    This Static class holds the hyper-parameters, model and experiment name information for the run.
    A Model class implementation must use the hyperparams via this Static class.
    """
    hyperparams_name = None
    hyperparams = None
    Model = None
    exp_name = None

    @classmethod
    def set_context(cls, hyperparams_name, exp_name=None):
        cls.hyperparams_name = hyperparams_name
        cls.exp_name = exp_name or cls.hyperparams.exp_name
        cls.hyperparams = HyperparamsFactory.get_hyperparams(cls.hyperparams_name)
        cls.Model = ModelFactory.get_model(cls.hyperparams.model)

    @classmethod
    def __repr__(cls):
        return "ExperimentContext: hyperparams_name: {}, exp_name: {}".format(cls.hyperparams_name, cls.exp_name)
