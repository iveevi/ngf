import numpy as np
import torch
import torch.nn.functional as functional

from dataclasses import dataclass

from mlp import MLP

@dataclass
class Sample:
    points: torch.Tensor
    features: torch.Tensor

class NGF:
    def __init__(self, points: torch.Tensor, complexes: torch.Tensor, features: torch.Tensor, config: dict, mlp=None):
        self.points = points
        self.complexes = complexes
        self.features = features

        self.encoding_levels = config['encoding_levels']
        self.jittering = config['jittering']
        self.encoder = config['encoder']
        self.normals = config['normals']

        features = config['features']
        ffin = features

        if self.encoder == 'sincos':
            ffin = features + 3 * (2 * self.encoding_levels + 1)
            self.encoder = self.sincos
        elif self.encoder == 'skewed':
            ffin = features + 3 * (self.encoding_levels + 1)
            self.encoder = self.skewed
        else:
            raise ValueError('Unknown encoder: %s' % self.encoder)

        self.mlp = MLP(ffin, functional.leaky_relu).cuda()
        if mlp is not None:
            self.mlp.load_state_dict(mlp.state_dict())

        self.config = config

    def eval(self, rate):
        sample = None
        if self.jittering:
            sample = self.sample_jittered(rate)
        else:
            sample = self.sample(rate)

        X = self.encoder(sample)
        return self.mlp(sample.points, X)

    def eval_uniform(self, rate):
        sample = self.sample(rate)
        X = self.encoder(sample)
        return self.mlp(sample.points, X)

    # Encoding functions
    def sincos(self, sample: Sample):
        bases    = sample.points
        features = sample.features

        X = [ features, bases ]
        for i in range(self.encoding_levels):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.cat(X, dim=-1)

        return X

    def skewed(self, sample: Sample):
        bases    = sample.points
        features = sample.features

        X = [ features, bases ]

        for i in range(self.encoding_levels):
            k = 2 ** (i/3.0)
            X += [ torch.sin(k * bases + 2/(i + 1)) ]

        X = torch.cat(X, dim=-1)

        return X

    # Functionals
    def eval_normals(self, rate: int):
        from functorch import jacfwd, jacrev, grad, vjp

        U = torch.linspace(0.0, 1.0, steps=rate).cuda()
        V = torch.linspace(0.0, 1.0, steps=rate).cuda()
        U, V = torch.meshgrid(U, V, indexing='ij')

        corner_points = self.points[self.complexes, :]
        corner_features = self.features[self.complexes, :]

        feature_size = self.features.shape[1]

        U, V = U.reshape(-1), V.reshape(-1)
        U = U.repeat((self.complexes.shape[0], 1))
        V = V.repeat((self.complexes.shape[0], 1))

        def ftn(U, V):
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

            sample = Sample(lerped_points, lerped_features)
            X = self.encoder(sample)
            return self.mlp(sample.points, X)

        delta = 1/rate
        Up, Um = U + delta, U - delta
        Vp, Vm = V + delta, V - delta

        v_Up, v_Um = ftn(Up, V), ftn(Um, V)
        v_Vp, v_Vm = ftn(U, Vp), ftn(U, Vm)

        v_dU = (v_Up - v_Um)/(2 * delta)
        v_dV = (v_Vp - v_Vm)/(2 * delta)

        N = torch.cross(v_dU, v_dV)
        length = torch.linalg.vector_norm(N, axis=1)

        return N/length.unsqueeze(-1)

    # Sampling functions
    def sample(self, rate: int) -> Sample:
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

        return Sample(lerped_points, lerped_features)

    def sample_jittered(self, rate: int) -> Sample:
        delta = 0.45/(rate - 1)

        U = torch.linspace(0.0, 1.0, steps=rate).cuda()
        V = torch.linspace(0.0, 1.0, steps=rate).cuda()
        U, V = torch.meshgrid(U, V, indexing='ij')

        corner_points = self.points[self.complexes, :]
        corner_features = self.features[self.complexes, :]

        feature_size = self.features.shape[1]

        U, V = U.reshape(-1), V.reshape(-1)
        U = U.repeat((self.complexes.shape[0], 1))
        V = V.repeat((self.complexes.shape[0], 1))

        U_plus, U_minus = U, (1.0 - U)
        V_plus, V_minus = V, (1.0 - V)

        UV_interior = (U_minus * U_plus * V_minus * V_plus) > 0

        rand_U = (2 * torch.rand(U.shape, device='cuda') - 1) * delta
        rand_V = (2 * torch.rand(V.shape, device='cuda') - 1) * delta

        rand_U *= UV_interior
        rand_V *= UV_interior

        jittered_U = U + rand_U
        jittered_V = V + rand_U

        U_plus, U_minus = jittered_U.unsqueeze(-1), (1.0 - jittered_U).unsqueeze(-1)
        V_plus, V_minus = jittered_V.unsqueeze(-1), (1.0 - jittered_V).unsqueeze(-1)

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

        return Sample(lerped_points, lerped_features)

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
    return NGF(data['points'], data['complexes'], data['features'], data['config'], mlp=data['model'])
