from torch import nn
from . import nets


class GNode(nn.Module):
    def __init__(self):
        super(GNode, self).__init__()
        self.gan = nets.ToyGAN('node')

    @property
    def name(self):
        return self.gan.name
