import numpy as np
import torch
import torch.nn.functional as F

from mlp import MLP

# TODO: show surface with encoding values... (like ENS)
class NGF:
    def __init__(self, points: torch.Tensor, complexes: torch.Tensor, features: torch.Tensor, config: dict):
        self.points = points
        self.complexes = complexes
        self.features = features
        self.encoding_levels = config['encoding_levels']
        self.config = config

        self.encoder = config['encoder']
        features = config['features']
        ffin = features

        if self.encoder == 'sincos':
            ffin = features + 3 * (2 * self.encoding_levels + 1)
            self.encoder = self.sincos
        elif self.encoder == 'skewed':
            ffin = features + 3 * (self.encoding_levels + 1)
            self.encoder = self.skewed
        elif self.encoder == 'extint':
            ffin = features + 4 * (self.encoding_levels + 1)
            self.encoder = self.extint
        else:
            raise ValueError('Unknown encoder: %s' % self.encoder)

        self.mlp = MLP(ffin, F.leaky_relu).cuda()

        # Compile the kernel
        self.sample_kernel = torch.compile(self.sample_kernel_py, fullgraph=True)

    def sample(self, rate):
        # TODO: use compiled version
        return self.sample_kernel_py(rate)

    def eval(self, rate):
        # TODO: revert to old return type
        X = self.sample(rate)
        base = X['points']
        X = self.encoder(**X)
        return self.mlp(base, X)

    # Encoding functions
    def sincos(self, **kwargs):
        bases    = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.encoding_levels):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        return X

    def skewed(self, **kwargs):
        bases    = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]

        for i in range(self.encoding_levels):
            k = 2 ** (i/2.0)
            X += [ torch.sin(k * bases + 2/(i + 1)) ]

        X = torch.cat(X, dim=-1)

        return X

    def extint(self, **kwargs):
        bases    = kwargs['points']
        features = kwargs['features']
        uvs      = kwargs['uvs']

        X = [ features, bases, uvs ]
        for i in range(self.encoding_levels):
            k = 2 ** (i/2.0)
            X += [ torch.sin(k * bases + 2/(i + 1)), torch.sin(k * uvs + 2/(i + 1)) ]

        X = torch.cat(X, dim=-1)

        return X

    def sample_kernel_py(self, rate):
        # TODO: custom lerper for features?
        U = torch.linspace(0.0, 1.0, steps=rate).cuda()
        V = torch.linspace(0.0, 1.0, steps=rate).cuda()
        U, V = torch.meshgrid(U, V, indexing='ij')

        corner_points = self.points[self.complexes, :]
        corner_features = self.features[self.complexes, :]

        feature_size = self.features.shape[1]

        U, V = U.reshape(-1), V.reshape(-1)
        U = U.repeat((self.complexes.shape[0], 1))
        V = V.repeat((self.complexes.shape[0], 1))

        U_plus, U_minus = U.unsqueeze(-1), (1.0 - U).unsqueeze(-1)
        V_plus, V_minus = V.unsqueeze(-1), (1.0 - V).unsqueeze(-1)

        lp00 = corner_points[:, 0, :].unsqueeze(1) * U_minus * V_minus
        lp01 = corner_points[:, 1, :].unsqueeze(1) * U_plus * V_minus
        lp10 = corner_points[:, 3, :].unsqueeze(1) * U_minus * V_plus
        lp11 = corner_points[:, 2, :].unsqueeze(1) * U_plus * V_plus

        lerped_points = (lp00 + lp01 + lp10 + lp11).reshape(-1, 3)

        lf00 = corner_features[:, 0, :].unsqueeze(1) * U_minus * V_minus
        lf01 = corner_features[:, 1, :].unsqueeze(1) * U_plus * V_minus
        lf10 = corner_features[:, 3, :].unsqueeze(1) * U_minus * V_plus
        lf11 = corner_features[:, 2, :].unsqueeze(1) * U_plus * V_plus

        lerped_features = (lf00 + lf01 + lf10 + lf11).reshape(-1, feature_size)

        UV = (16 * U_minus * V_minus * U_plus * V_plus).square().reshape(-1, 1)

        return { 'points' : lerped_points, 'features' : lerped_features, 'uvs' : UV }

    def save(self, filename):
        model = {
            'model'     : self.mlp,
            'features'  : self.features,
            'complexes' : self.complexes,
            'points'    : self.points,
            'config'    : self.config
        }

        torch.save(model, filename)

def load_ngf(data):
    return NGF(data['points'], data['complexes'], data['features'], data['config'])
