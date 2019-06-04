
import numpy as np
import typing

#-----------------------------
# Buffer class 
# -------------------------
class Buffer(object):
    """
    A finite-memory buffer that rewrites oldest data when buffer is full.
    Stores tuples of the form (feature, action, reward, next feature). 
    """
    def __init__(self, size=50000):
        self.size = size
        self.buffer = []
        self.next_idx = 0

    def add(self, data):
        if self.next_idx == len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.size

    def sample(self, batch_size=1):
        idxs = np.random.randint(len(self.buffer), size=batch_size)
        return [self.buffer[i] for i in idxs]


import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dimensions):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dimensions)-1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i+1]))

    def forward(self, x):
        for l in self.layers[:-1]:
            x = nn.functional.relu(l(x))
        x = self.layers[-1](x)
        return x


    def initialize(self):
        '''
        Initialize weights using Glorot initialization 
        or also known as Xavier initialization
        '''
        def initialize_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.apply(initialize_weights)

class DQNWithPrior(nn.Module):
    def __init__(self,dimensions,scale=5):
        '''
        :param dimensions: dimensions of the neural network
        prior network with immutable weights and
        difference network whose weights will be learnt
        '''

        super(DQNWithPrior,self).__init__()
        self.f_diff = MLP(dimensions)
        self.f_prior = MLP(dimensions)
        self.scale = scale
    def forward(self, x):
        '''
        :param x: input to the network
        :return: computes f_diff(x) + f_prior(x)
        performs forward pass of the network
        '''
        return self.f_diff(x) + self.scale*self.f_prior(x)

    def initialize(self):
        '''
        :param scale: scale with which weights need to be initialized
        Initialize weights using Glorot initialization and freeze f_prior
        or also known as Xavier initialization
        '''
        self.f_prior.initialize()
        self.f_prior.eval()
        self.f_diff.initialize()

    def parameters(self, recurse:bool =True):
        '''
        :param recurse: bool Recursive or not
        :return: all the parameters of the network that are mutable or learnable
        '''
        return  self.f_diff.parameters(recurse)