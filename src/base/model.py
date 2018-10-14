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
                init.constant(module.bias, 0.001)

        self.apply(init_fn)
