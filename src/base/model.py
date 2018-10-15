from __future__ import print_function
from torch import nn
from torch.nn import init

n_batch_logits = 32


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @classmethod
    def copy(cls, model):
        module = cls()
        module.load_state_dict(model.state_dict())
        return module

    def init_params(self):
        def init_fn(module):
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
                init.constant_(module.bias, 0.001)

        self.apply(init_fn)


class BaseGan(BaseModel):
    def __init__(self):
        super(BaseGan, self).__init__()

    def step_train_generator(self, z):
        return NotImplementedError

    def step_train_discriminator(self, x, z):
        return NotImplementedError

    def step_train_autoencoder(self, x, z):
        return NotImplementedError
