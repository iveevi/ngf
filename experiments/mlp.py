import torch
import torch.nn as nn
import torch.nn.functional as F

POINT_ENCODING_SIZE = 10

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
        encodings = kwargs['encodings']

        X = torch.cat([ encodings, bases ], dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)

        return bases + X

class MLP_Positional_Encoding(nn.Module):
    # TODO: variable levels..
    L = 8

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

    def forward(self, **kwargs):
        bases = kwargs['points']
        encodings = kwargs['encodings']

        X = [ encodings, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)

        return bases + X

class MLP_Feature_Sinusoidal_Encoding(nn.Module):
    L = 8

    def __init__(self) -> None:
        super(MLP_Feature_Sinusoidal_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear((2 * self.L + 1) * POINT_ENCODING_SIZE + 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        encodings = kwargs['encodings']

        X = [ encodings ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * encodings), torch.cos(2 ** i * encodings) ]
        X += [ bases ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)

        return bases + X

class MLP_Feature_Position_Encoding(nn.Module):
    L = 8

    def __init__(self) -> None:
        super(MLP_Feature_Position_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear((2 * self.L + 1) * (POINT_ENCODING_SIZE + 3), 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        encodings = kwargs['encodings']

        X = [ encodings, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * encodings), torch.cos(2 ** i * encodings) ]
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)

        return bases + X

# TODO: use UV output... (local positional encoding, forget about boundaries for now...)
class MLP_UV(nn.Module):
    def __init__(self) -> None:
        super(MLP_UV, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + 2, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        encodings = kwargs['encodings']
        uvs = kwargs['uv']

        # TODO: operate on dict of tensors
        X = torch.cat([ encodings, uvs ], dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)

        return bases + X

class MLP_UV_Sinusoidal_Encoding(nn.Module):
    L = 8

    def __init__(self) -> None:
        super(MLP_UV_Sinusoidal_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 2, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        encodings = kwargs['encodings']
        uvs = kwargs['uv']

        # TODO: operate on dict of tensors
        X = [ encodings, uvs ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * uvs), torch.cos(2 ** i * uvs) ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)

        return bases + X

class MLP_Feature_UV_Sinusoidal_Encoding(nn.Module):
    L = 8

    def __init__(self) -> None:
        super(MLP_Feature_UV_Sinusoidal_Encoding, self).__init__()

        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear((2 * self.L + 1) * (2 + POINT_ENCODING_SIZE), 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, **kwargs):
        bases = kwargs['points']
        encodings = kwargs['encodings']
        uvs = kwargs['uv']

        # TODO: operate on dict of tensors
        X = [ encodings, uvs ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * encodings), torch.cos(2 ** i * encodings) ]
            X += [ torch.sin(2 ** i * uvs), torch.cos(2 ** i * uvs) ]
        X = torch.cat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)

        return bases + X

# TODO: try a morlet activation on the Uv pos enc...
