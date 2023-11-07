import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POINT_ENCODING_SIZE = 20

class MLP_Simple(nn.Module):
    def __init__(self) -> None:
        super(MLP_Simple , self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = torch.cat([ features, bases ], dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)

        return bases + X

class MLP_Positional_Encoding(nn.Module):
    L = 10

    def __init__(self) -> None:
        super(MLP_Positional_Encoding, self).__init__()

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

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = F.relu((self.s0 if i == 0 else self.s1) * X)

        return bases + X

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

        self.s0 = nn.Parameter(torch.tensor(1.0))
        self.s1 = nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X * (self.s0 if i == 0 else self.s1))

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

        self.s0 = nn.Parameter(torch.tensor(1.0))
        self.s1 = nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.exp(-X ** 2/(self.s0 if i == 0 else self.s1))

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

        self.s0 = nn.Parameter(torch.tensor(1.0))
        self.s1 = nn.Parameter(torch.tensor(1.0))

        morlet = lambda x, s: torch.exp(-x ** 2/s) * torch.sin(s * x)
        self.morlet = torch.compile(morlet, mode='reduce-overhead')

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = self.morlet(X, self.s0 if i == 0 else self.s1)

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

        self.onion_kernel = torch.compile(self.onion, mode='reduce-overhead')

    def onion(self, x, s):
        softplus = torch.log(1 + torch.exp(s * x))
        sin = torch.sin(s * x)
        exp = torch.exp(-x ** 2/s)
        return softplus * sin * exp

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = self.onion_kernel(X, self.s0 if i == 0 else self.s1)

        return bases + X

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

        self.s0 = nn.Parameter(torch.tensor(1.0))
        self.s1 = nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sinc((self.s0 if i == 0 else self.s1) * X)

        return bases + X

class MLP_Positional_Rexin_Encoding(nn.Module):
    L = 10

    def __init__(self) -> None:
        super(MLP_Positional_Rexin_Encoding, self).__init__()

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

        # Rexin activation (ReLU, Exponential, Sine)
        self.rexin = lambda x, s: torch.sinc(s * x) + torch.log(1 + torch.exp(x))
        self.rexin = torch.compile(self.rexin, mode='reduce-overhead')

    def forward(self, **kwargs):
        bases = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = self.rexin(X, (self.s0 if i == 0 else self.s1))

        return bases + X
