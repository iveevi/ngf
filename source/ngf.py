from __future__ import annotations

import meshio
import logging
import numpy as np
import torch
import torch.nn as nn

from typing import Callable

# from util import SIREN


# TODO: try a SIREN network...
class MLP(nn.Module):
    def __init__(self, ffin: int) -> None:
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(ffin, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.layers(x)

    def stream(self):
        weights = [layer.weight.data.cpu() for layer in self.layers if isinstance(layer, nn.Linear)]
        biases = [layer.bias.data.cpu() for layer in self.layers if isinstance(layer, nn.Linear)]

        bytestream = b''
        for w in weights:
            bytestream += w.shape[0].to_bytes(4, 'little')
            bytestream += w.shape[1].to_bytes(4, 'little')
            bytestream += w.numpy().astype('float32').tobytes()

        for b in biases:
            bytestream += b.shape[0].to_bytes(4, 'little')
            bytestream += b.numpy().astype('float32').tobytes()

        return bytestream


def spherical_to_cartesian(spherical):
    radius = spherical[..., 0]
    theta = spherical[..., 1]
    phi = spherical[..., 2]
    x = theta.cos() * phi.sin()
    y = theta.sin() * phi.sin()
    z = phi.cos()
    return radius.unsqueeze(-1) * torch.stack((x, y, z), dim=-1)


@torch.jit.script
def triangle_wave(x):
    return (2 * x.fmod(1) - 1).abs()


# Positional encoding
@torch.jit.script
def positional_encoding(vector: torch.Tensor, levels: int) -> list[torch.Tensor]:
    result = []
    for i in range(levels):
        k = 2 ** i
        result += [torch.sin(k * vector), torch.cos(k * vector)]
    return result


class NGF:
    def __init__(self,
                 points: torch.Tensor,
                 features: torch.Tensor,
                 complexes: torch.Tensor,
                 fflevels: int,
                 jittering: bool,
                 normals: bool,
                 mlp=None) -> None:
        self.points = points
        self.features = features
        self.complexes = complexes
        self.fflevels = fflevels
        self.jittering = jittering
        self.normals = normals

        self.ffin = self.features.shape[-1] + 3
        self.mlp = MLP(self.ffin).cuda()
        if mlp is not None:
            self.mlp.load_state_dict(mlp.state_dict())

        self.sampler = self.sample_uniform
        if self.jittering:
            self.sampler = self.sample_jittered

        # Caches
        self.uv_cache = {}
        self.uv_mask = {}

        # Log details
        logging.info('Instantiated neural geometry field with properties:')
        logging.info(f'     Vertex count: {self.points.shape[0]}')
        logging.info(f'     Quad count:   {self.complexes.shape[0]}')
        logging.info(f'     Feature size: {self.features.shape[-1]}')
        logging.info(f'     FF levels:    {fflevels}')
        logging.info(f'     Jittering:    {jittering}')
        logging.info(f'     Normals:      {normals}')

    # List of parameters
    def parameters(self):
        return list(self.mlp.parameters()) + [self.points, self.features]

    # Number of patches
    def patches(self) -> int:
        return self.complexes.shape[0]

    @staticmethod
    def interpolate(points, features, complexes, U, V):
        fsize = features.shape[1]
        corner_points = points[complexes]
        corner_features = features[complexes]

        Up, Um = U.unsqueeze(-1), (1.0 - U).unsqueeze(-1)
        Vp, Vm = V.unsqueeze(-1), (1.0 - V).unsqueeze(-1)

        lp00 = corner_points[:, 0, :].unsqueeze(1) * Um * Vm
        lp01 = corner_points[:, 1, :].unsqueeze(1) * Up * Vm
        lp10 = corner_points[:, 3, :].unsqueeze(1) * Um * Vp
        lp11 = corner_points[:, 2, :].unsqueeze(1) * Up * Vp

        lp = (lp00 + lp01 + lp10 + lp11).reshape(-1, 3)

        lf00 = corner_features[:, 0, :].unsqueeze(1) * Um * Vm
        lf01 = corner_features[:, 1, :].unsqueeze(1) * Up * Vm
        lf10 = corner_features[:, 3, :].unsqueeze(1) * Um * Vp
        lf11 = corner_features[:, 2, :].unsqueeze(1) * Up * Vp

        lf = (lf00 + lf01 + lf10 + lf11).reshape(-1, fsize)

        return lp, lf

    # Extraction methods
    # TODO: interpolation kernel
    def eval(self, U, V):
        lp, lf = NGF.interpolate(self.points, self.features, self.complexes, U, V)
        lin = torch.cat((lp, lf), dim=-1)
        return lp + self.mlp(lin)

    def base(self, rate):
        U, V = self.sample_uniform(rate)

        corner_points = self.points[self.complexes, :]

        Up, Um = U.unsqueeze(-1), (1.0 - U).unsqueeze(-1)
        Vp, m = V.unsqueeze(-1), (1.0 - V).unsqueeze(-1)

        lp00 = corner_points[:, 0, :].unsqueeze(1) * Um * m
        lp01 = corner_points[:, 1, :].unsqueeze(1) * Up * m
        lp10 = corner_points[:, 3, :].unsqueeze(1) * Um * Vp
        lp11 = corner_points[:, 2, :].unsqueeze(1) * Up * Vp

        return (lp00 + lp01 + lp10 + lp11).reshape(-1, 3)

    # Sampling functions
    def sample_uniform(self, rate: int):
        if rate in self.uv_cache:
            return self.uv_cache[rate]

        U = torch.linspace(0.0, 1.0, steps=rate, device='cuda')
        V = torch.linspace(0.0, 1.0, steps=rate, device='cuda')
        U, V = torch.meshgrid(U, V, indexing='ij')

        U, V = U.reshape(-1), V.reshape(-1)
        U = U.repeat((self.complexes.shape[0], 1))
        V = V.repeat((self.complexes.shape[0], 1))

        self.uv_cache[rate] = (U, V)

        return U, V

    def sample_jittered(self, rate: int):
        U, V = self.sample_uniform(rate)

        if rate in self.uv_mask:
            UV_interior = self.uv_mask[rate]
        else:
            U_plus, U_minus = U, (1.0 - U)
            V_plus, V_minus = V, (1.0 - V)
            UV_interior = (U_minus * U_plus * V_minus * V_plus) > 0
            self.uv_mask[rate] = UV_interior

        delta = 0.45/(rate - 1)

        rtheta = 2 * np.pi * torch.rand(*U.shape, device='cuda')
        rr = torch.rand(*U.shape, device='cuda').sqrt()
        ru = delta * rr * rtheta.cos() * UV_interior
        rv = delta * rr * rtheta.sin() * UV_interior

        return U + ru, V + rv

    def save(self, filename):
        """Save into a PyTorch (PT) file"""
        torch.save({
            'points': self.points,
            'features': self.features,
            'complexes': self.complexes,
            'fflevels': self.fflevels,
            'jittering': self.jittering,
            'normals': self.normals,
            'model': self.mlp,
        }, filename)

    def stream(self):
        """Convert neural geometry field to byte stream"""
        # TODO: quantization

        sizes = [self.complexes.shape[0], self.points.shape[0], self.features.shape[-1]]
        size_bytes = np.array(sizes, dtype=np.int32).tobytes()

        with torch.no_grad():
            points_bytes = self.points.cpu().numpy().tobytes()
            features_bytes = self.features.cpu().numpy().tobytes()
            complexes_bytes = self.complexes.cpu().numpy().tobytes()

        mlp_bytes = self.mlp.stream()

        return size_bytes + points_bytes + features_bytes + complexes_bytes + mlp_bytes

    @staticmethod
    def from_base(path: str, normalizer: Callable, features: int, config: dict = dict()) -> NGF:
        mesh = meshio.read(path)
        points = torch.from_numpy(mesh.points)
        complexes = torch.from_numpy(mesh.cells_dict['quad'])

        points = normalizer(points.float().cuda())
        features = torch.zeros((points.shape[0], features)).cuda()
        complexes = complexes.int().cuda()

        points.requires_grad = True
        features.requires_grad = True

        return NGF(points, features, complexes,
                   config.setdefault('fflevels', 8),
                   config.setdefault('jittering', True),
                   config.setdefault('normals', True))

    @staticmethod
    def from_pt(path: str) -> NGF:
        data = torch.load(path)
        return NGF(data['points'],
                   data['features'],
                   data['complexes'],
                   data['fflevels'],
                   data['jittering'],
                   data['normals'],
                   mlp=data['model'])
