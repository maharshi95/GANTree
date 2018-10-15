from base.model import BaseGan
from base.dataloader import BaseDataLoader


class BaseTrainer(object):
    def __init__(self, model, data_loader, n_iterations):
        # type: (BaseGan, BaseDataLoader, int) -> None
        self.model = model
        self.data_loader = data_loader
        self.n_iterations = n_iterations

    def train(self):
        return NotImplementedError
