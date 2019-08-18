import torch as tr
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter

from base.model import BaseModel
from utils.decorators import make_tensor


class ConvBlock(nn.Sequential):
    def __init__(self, in_filters, out_filters, batch_norm=True, kernel_size=3, stride=2, leak=0.2):
        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_filters, 0.8))
        layers.append(nn.LeakyReLU(leak))

        super(ConvBlock, self).__init__(*layers)


class Net(BaseModel):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.init_params()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

    def predict(self, x):
        log_softmax = self.forward(x)
        return tr.argmax(log_softmax, dim=-1)


class MNISTCritic(BaseModel):
    def __init__(self, n_classes=10):
        super(MNISTCritic, self).__init__()

        self.n_classes = n_classes

        self.conv1 = ConvBlock(1, 32, batch_norm=False, kernel_size=5, stride=2)
        self.conv2 = ConvBlock(32, 64, batch_norm=False, kernel_size=5, stride=2)
        self.conv3 = ConvBlock(64, 64, batch_norm=False, kernel_size=5, stride=2)
        self.conv4 = ConvBlock(64, 64, batch_norm=True, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 4 * 4, n_classes)
        self.init_params()

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2))
        x = self.conv1(x)

        x = F.pad(x, (2, 2, 2, 2))
        x = self.conv2(x)

        x = F.pad(x, (2, 2, 2, 2))
        x = self.conv3(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.conv4(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.8, training=self.training)
        x = F.log_softmax(x, dim=-1)
        return x

    @make_tensor()
    def predict(self, x):
        with tr.no_grad():
            log_softmax = self.forward(x)
            return tr.argmax(log_softmax, dim=-1)

    @make_tensor()
    def probs(self, x):
        with tr.no_grad():
            log_softmax = self.forward(x)
            return tr.exp(log_softmax)
