# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import torch
import torch.nn.functional as F


def sigmoid_activation(x):
    return torch.sigmoid(5 * x)


def tanh_activation(x):
    return torch.tanh(2.5 * x)


def abs_activation(x):
    return torch.abs(x)


def gauss_activation(x):
    return torch.exp(-5.0 * x**2)


def identity_activation(x):
    return x


def sin_activation(x):
    return torch.sin(x)


def relu_activation(x):
    return F.relu(x)

def elu_activation(z):
    # return z if z > 0.0 else math.exp(z) - 1
    return torch.where(z>0.0, z, torch.exp(z) - 1)

def lelu_activation(z):
    leaky = 0.005
    # return z if z > 0.0 else leaky * z
    return torch.where(z>0.0, z, leaky*z)

def selu_activation(z):
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    # return lam * z if z > 0.0 else lam * alpha * (math.exp(z) - 1)
    return torch.where(z>0.0, lam * z, lam * alpha * (torch.exp(z) - 1))

def softplus_activation(z):
    # z = max(-60.0, min(60.0, 5.0 * z))
    # return 0.2 * math.log(1 + math.exp(z))
    z = torch.max(torch.tensor(-60.0), torch.min(torch.tensor(60.0), 5.0 * z))
    return 0.2 * torch.log(1 + torch.exp(z))

def clamped_activation(z):
    return torch.max(torch.tensor(-1.0), torch.min(torch.tensor(1.0), z))

def inv_activation(z):
    return z.pow(-1) # Seems like a bad idea since 1/0 = inf
    # I recommend avoinding this one if possible?

def log_activation(z):
    z = torch.max(torch.tensor(1e-7), z)
    return torch.log(z)

def exp_activation(z):
    z = torch.max(torch.tensor(-60.0), torch.min(torch.tensor(60.0), z))
    return torch.exp(z)

def hat_activation(z):
    return torch.max(torch.tensor(0.0), 1 - torch.abs(z))

def square_activation(z):
    return z ** 2


def cube_activation(z):
    return z ** 3

str_to_activation = {
    'sigmoid': sigmoid_activation,
    'tanh': tanh_activation,
    'abs': abs_activation,
    'gauss': gauss_activation,
    'identity': identity_activation,
    'sin': sin_activation,
    'relu': relu_activation,
    'elu' : elu_activation,
    'lelu' : lelu_activation,
    'selu' : selu_activation,
    'softplus': softplus_activation,
    'clamped':clamped_activation,
    'inv':inv_activation,
    'log':log_activation,
    'exp':exp_activation,
    'hat':hat_activation,
    'square':square_activation,
    'cube':cube_activation,

}
