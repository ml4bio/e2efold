from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import argparse
import logging
import numpy as np
import torch
import torch.nn as nn

t_float = torch.float32
np_float = np.float32
str_float = "float32"

# opts = argparse.ArgumentParser(description='gpu option')
# opts.add_argument('-gpu', type=int, default=1, help='-1: cpu; 0 - ?: specific gpu index')

# args, _ = opts.parse_known_args()
# if torch.cuda.is_available() and args.gpu >= 0:
#     DEVICE = torch.device('cuda:' + str(args.gpu))
#     print('use gpu indexed: %d' % args.gpu)
# else:
#     DEVICE = torch.device('cpu')
#     print('use cpu')


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def soft_sign(x, k):
    return torch.sigmoid(k * x)

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "softmax": nn.Softmax(dim=-1),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}
