import torch as tr
from torch import nn
from torch.nn import functional as F
import numpy as np
from base.hyperparams import Hyperparams
from exp_context import ExperimentContext
from base.model import BaseModel
from torch.autograd import Variable

H = ExperimentContext.Hyperparams  # type: Hyperparams


def reparameterization(mu, logvar, z_dim):
    std = tr.exp(logvar / 2)
    sampled_z = Variable(tr.Tensor(np.random.normal(0, 1, (mu.size(0), z_dim)))).cuda()
    z = sampled_z * std + mu
    return z

class Generator(BaseModel):
    # initializers
    def __init__(self, z_dim, channel, d=128):
        super(Generator, self).__init__()

        self.z_dim = z_dim

        self.deconv1 = nn.ConvTranspose2d(z_dim, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, channel, 4, 2, 1)

        self.init_params()

    # forward method
    def forward(self, x):
        x = F.relu(self.deconv1_bn(self.deconv1(x.view(-1, self.z_dim, 1, 1))))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

class Encoder(BaseModel):
    # initializers
    def __init__(self, z_dim, channel, d=128):
        super(Encoder, self).__init__()
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(channel, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5_mu = nn.Conv2d(d*8, z_dim, 4, 1, 0)
        self.conv5_var = nn.Conv2d(d*8, z_dim, 4, 1, 0)

        self.init_params()

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)

        mu = self.conv5_mu(x)
        var = self.conv5_var(x)
        
        z = reparameterization(mu.view(-1, self.z_dim), var.view(-1, self.z_dim), self.z_dim)
    
        return z

class Discriminator(BaseModel):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()

        self.z_dim = z_dim
        
        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.activation = nn.Sigmoid()

        self.init_params()

    def forward(self, z):
        z = F.leaky_relu(self.fc1(z), 0.2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = self.fc3(z)
        
        validity = self.activation(z)
        
        return validity