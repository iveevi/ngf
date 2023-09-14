import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POINT_ENCODING_SIZE = 15
MATRIX_SIZE = 0
ENCODING_SIZE = POINT_ENCODING_SIZE + MATRIX_SIZE * 3

# NSC subdivision complex indices
def indices(N):
    gim_indices = []
    for i in range(N - 1):
        for j in range(N - 1):
            gim_indices.append([i * N + j, (i + 1) * N + j, i * N + j + 1])
            gim_indices.append([(i + 1) * N + j, (i + 1) * N + j + 1, i * N + j + 1])
    return np.array(gim_indices).reshape(-1, 3).astype(np.int32)

def quad_indices(N):
    gim_indices = []
    for i in range(N - 1):
        for j in range(N - 1):
            gim_indices.append([i * N + j, i * N + j + 1, (i + 1) * N + j + 1, (i + 1) * N + j])
    return np.array(gim_indices).reshape(-1, 4).astype(np.int32)

# Lipschitz normalization
class LipschitzNormalization(nn.Module):
    one = torch.tensor(1.0).cuda()

    def __init__(self, c):
        super(LipschitzNormalization, self).__init__()
        self.c = torch.nn.Parameter(torch.tensor(c))

    def forward(self, W):
        # Normalize W to have Lipschitz bound of at most c
        absrowsum = torch.sum(torch.abs(W), dim=1)
        softplus_c = torch.nn.functional.softplus(self.c)
        scale = torch.min(LipschitzNormalization.one, softplus_c/absrowsum)
        return W * scale.unsqueeze(1)

L = 0
class NSC(nn.Module):
    def __init__(self) -> None:
        super(NSC, self).__init__()

        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (L + 1) * 3, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 3),
            # nn.Linear(128, MATRIX_SIZE)
        ]

        c0 = torch.pow(torch.tensor(0.001), 1.0/len(self.encoding_linears))
        c0 = torch.log(torch.exp(c0) - LipschitzNormalization.one)
        self.regularizers = [
            LipschitzNormalization(c0),
            LipschitzNormalization(c0),
            LipschitzNormalization(c0),
            LipschitzNormalization(c0),
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)
        self.regularizers = nn.ModuleList(self.regularizers)

    def forward(self, args):
        points = args['points']
        encodings = args['encodings']
        complexes = args['complexes']
        resolution = args['resolution']

        local_points = points[complexes, :]
        local_encodings = encodings[complexes, :]

        local_points = local_points.reshape(-1, 2, 2, 3).permute(0, 3, 1, 2)
        p = F.interpolate(local_points, size=(resolution, resolution), mode='bilinear', align_corners=True)
        p = p.permute(0, 2, 3, 1)

        local_encodings = local_encodings.reshape(-1, 2, 2, ENCODING_SIZE).permute(0, 3, 1, 2)
        e = F.interpolate(local_encodings, size=(resolution, resolution), mode='bilinear', align_corners=True)
        e = e.permute(0, 2, 3, 1)

        # Now we can evaluate using the network
        X = []
        if POINT_ENCODING_SIZE > 0:
            enc = e[:, :, :, :POINT_ENCODING_SIZE]
            X = [ enc ]

        for i in range(L + 1):
            X.append(torch.sin(2 ** i * p))

        # X = torch.concat((p, torch.sin(p), enc), dim=-1)
        X = torch.concat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            W, b = linear.weight, linear.bias

            X = torch.matmul(X, W.t()) + b
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)

        if MATRIX_SIZE > 0:
            X = torch.sin(X)
            m = e[:, :, :, POINT_ENCODING_SIZE:]
            m = m.reshape(-1, resolution, resolution, 3, MATRIX_SIZE)
            X = torch.matmul(m, X.unsqueeze(-1)).squeeze(-1)

        lipschitz = LipschitzNormalization.one.clone()
        return p + X, X, lipschitz

class NSubComplex(nn.Module):
    def __init__(self) -> None:
        super(NSubComplex , self).__init__()

        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (L + 1) * 3, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        c0 = torch.pow(torch.tensor(0.1), 1.0/len(self.encoding_linears))
        c0 = torch.log(torch.exp(c0) - LipschitzNormalization.one)
        self.regularizers = [
            LipschitzNormalization(c0),
            LipschitzNormalization(c0),
            LipschitzNormalization(c0),
            LipschitzNormalization(c0),
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)
        self.regularizers = nn.ModuleList(self.regularizers)

    def forward(self, bases, encodings):
        enc = encodings

        X = [ enc ]
        for i in range(L + 1):
            X.append(torch.sin(2 ** i * bases))
        X = torch.concat(X, dim=-1)

        for i, (linear, regularizer) in enumerate(zip(self.encoding_linears, self.regularizers)):
            W, b = linear.weight, linear.bias
            # W = regularizer(W)

            X = torch.matmul(X, W.t()) + b
            if i < len(self.encoding_linears) - 1:
                X = F.elu(X)

        return bases + X

