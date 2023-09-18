import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POINT_ENCODING_SIZE = 20

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

def one_blob(s, t, sigma=0.1, k=32):
    t = t.repeat(s.shape[0], 1)
    s = s.unsqueeze(1)
    kernel = torch.exp(-((s - t) ** 2) / (2 * sigma ** 2))
    return kernel

class NSubComplex(nn.Module):
    L = 8
    K = 16

    def __init__(self) -> None:
        super(NSubComplex , self).__init__()

        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3 + (2 + 2 * self.K), 64),
            # nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3 + 2, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)
        self.T = torch.linspace(0, 1, self.K).cuda()

    def forward(self, bases, normals, encodings):
        X = [ encodings, bases ]
        for i in range(1, self.L + 1):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X += [ normals, one_blob(normals[:, 0], self.T), one_blob(normals[:, 1], self.T) ]
        # X += [ normals ] # , one_blob(normals[:, 0], self.T), one_blob(normals[:, 1], self.T) ]
        X = torch.concat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)

        return bases + X
