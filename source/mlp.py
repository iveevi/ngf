import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable

class MLP(nn.Module):
    def __init__(self, ffin: int, activation: Callable) -> None:
        super(MLP, self).__init__()

        # print('MLP: ffin =', ffin)

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            # nn.Linear(ffin, 64, dtype=torch.float16),
            # nn.Linear(64, 64,   dtype=torch.float16),
            # nn.Linear(64, 64,   dtype=torch.float16),
            # nn.Linear(64, 3,    dtype=torch.float16)
            
            nn.Linear(ffin, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)
        self.activation = activation

    def forward(self, bases, X):
        # X = X.to(torch.float16)

        X = self.encoding_linears[0](X)
        X = self.activation(X)

        X = self.encoding_linears[1](X)
        X = self.activation(X)

        X = self.encoding_linears[2](X)
        X = self.activation(X)

        X = self.encoding_linears[3](X)

        return bases + X

POINT_ENCODING_SIZE = 20

class MLP_Positional_ReLU_Encoding(nn.Module):
    L = 10

    def __init__(self) -> None:
        super(MLP_Positional_ReLU_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        X = self.encoding_linears[0](X)
        X = F.relu(X)
        X = self.encoding_linears[1](X)
        X = F.relu(X)
        X = self.encoding_linears[2](X)
        X = F.relu(X)
        X = self.encoding_linears[3](X)

        return bases + X

class MLP_Positional_LeakyReLU_Encoding(nn.Module):
    L = 10

    def __init__(self) -> None:
        super(MLP_Positional_LeakyReLU_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        X = self.encoding_linears[0](X)
        X = F.leaky_relu(X)
        X = self.encoding_linears[1](X)
        X = F.leaky_relu(X)
        X = self.encoding_linears[2](X)
        X = F.leaky_relu(X)
        X = self.encoding_linears[3](X)

        return bases + X

class MLP_Positional_Elu_Encoding(nn.Module):
    L = 10

    def __init__(self) -> None:
        super(MLP_Positional_Elu_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        X = self.encoding_linears[0](X)
        X = F.elu(X)
        X = self.encoding_linears[1](X)
        X = F.elu(X)
        X = self.encoding_linears[2](X)
        X = F.elu(X)
        X = self.encoding_linears[3](X)

        return bases + X

class MLP_Positional_LeakyElu_Encoding(nn.Module):
    L = 10

    def __init__(self) -> None:
        super(MLP_Positional_LeakyElu_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

        self.lelu = lambda x: 0.1 * x + 0.9 * torch.log(1 + torch.exp(x))
        self.lelu = torch.compile(self.lelu, mode='reduce-overhead')

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        X = self.encoding_linears[0](X)
        X = F.elu(X)
        X = self.encoding_linears[1](X)
        X = F.elu(X)
        X = self.encoding_linears[2](X)
        X = F.elu(X)
        X = self.encoding_linears[3](X)

        return bases + X

    # Avoid pickling the compiled function
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['lelu']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lelu = lambda x: 0.1 * x + 0.9 * torch.log(1 + torch.exp(x))
        self.lelu = torch.compile(self.lelu, mode='reduce-overhead')

class MLP_Positional_Siren_Encoding(nn.Module):
    L = 10

    def __init__(self) -> None:
        super(MLP_Positional_Siren_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        X = self.encoding_linears[0](X)
        X = torch.sin(X)
        X = self.encoding_linears[1](X)
        X = torch.sin(X)
        X = self.encoding_linears[2](X)
        X = torch.sin(X)
        X = self.encoding_linears[3](X)

        return bases + X

class MLP_Positional_Gaussian_Encoding(nn.Module):
    L = 10

    def __init__(self) -> None:
        super(MLP_Positional_Gaussian_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        X = self.encoding_linears[0](X)
        X = torch.exp(-X ** 2/2)
        X = self.encoding_linears[1](X)
        X = torch.exp(-X ** 2/2)
        X = self.encoding_linears[2](X)
        X = torch.exp(-X ** 2/2)
        X = self.encoding_linears[3](X)

        return bases + X

class MLP_Positional_Morlet_Encoding(nn.Module):
    L = 10

    def __init__(self) -> None:
        super(MLP_Positional_Morlet_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

        self.morlet = lambda x: torch.exp(-x ** 2) * torch.sin(x)
        self.morlet = torch.compile(self.morlet, mode='reduce-overhead')

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        X = self.encoding_linears[0](X)
        X = self.morlet(X)
        X = self.encoding_linears[1](X)
        X = self.morlet(X)
        X = self.encoding_linears[2](X)
        X = self.morlet(X)
        X = self.encoding_linears[3](X)

        return bases + X

    # Avoid pickling the compiled function
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['morlet']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.morlet = lambda x: torch.exp(-x ** 2) * torch.sin(x)
        self.morlet = torch.compile(self.morlet, mode='reduce-overhead')

class MLP_Positional_Sinc_Encoding(nn.Module):
    L = 10

    def __init__(self) -> None:
        super(MLP_Positional_Sinc_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        X = self.encoding_linears[0](X)
        X = torch.sinc(X)
        X = self.encoding_linears[1](X)
        X = torch.sinc(X)
        X = self.encoding_linears[2](X)
        X = torch.sinc(X)
        X = self.encoding_linears[3](X)

        return bases + X

class MLP_Positional_Onion_Encoding(nn.Module):
    L = 10

    def __init__(self) -> None:
        super(MLP_Positional_Onion_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

        self.s0 = nn.Parameter(torch.tensor(1.0))
        self.s1 = nn.Parameter(torch.tensor(1.0))
        self.s2 = nn.Parameter(torch.tensor(1.0))

        self.onion = lambda x: torch.log(1 + torch.exp(x)) * torch.sin(x) * torch.exp(-x ** 2)
        self.onion = torch.compile(self.onion, mode='reduce-overhead')

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        X = self.encoding_linears[0](X)
        X = self.onion(X)
        X = self.encoding_linears[1](X)
        X = self.onion(X)
        X = self.encoding_linears[2](X)
        X = self.onion(X)
        X = self.encoding_linears[3](X)

        return bases + X

    # Avoid pickling the compiled function
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['onion']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.onion = lambda x: torch.log(1 + torch.exp(x)) * torch.sin(x) * torch.exp(-x ** 2)
        self.onion = torch.compile(self.onion, mode='reduce-overhead')
