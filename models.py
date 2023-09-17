import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POINT_ENCODING_SIZE = 20
COMPLEX_ENCODING_SIZE = 20

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

class NSubComplex(nn.Module):
    L = 8
    Lpoint = 0
    Lcomplex = 0

    def __init__(self) -> None:
        super(NSubComplex , self).__init__()

        self.encoding_linears = [
            # nn.Linear(COMPLEX_ENCODING_SIZE * (self.Lcomplex + 1) + 3, 128),
            nn.Linear(POINT_ENCODING_SIZE + COMPLEX_ENCODING_SIZE + (2 * self.L + 1) * 3, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

        # displacements = torch.ones((self.L + 1, 3))
        # for i in range(self.L + 1):
        #     displacements[i] *= 1/2 ** i
        #
        # self.displacements = nn.Parameter(displacements)

    def forward(self, bases, encodings, complex_encodings):
        # X = []
        # for i in range(self.L + 1):
        #     X.append(torch.concat((torch.sin(2 ** i * encodings), complex_encodings, bases), dim=-1))
        #     # X.append(torch.concat((torch.sin(2 ** i * encodings), torch.sin(2 ** i * complex_encodings), bases), dim=-1))
        #
        # X = torch.stack(X)
        # # print(X.shape)

        X = [ encodings, complex_encodings, bases ]
        for i in range(1, self.L + 1):
            X.append(torch.sin(2 ** i * bases))
            X.append(torch.cos(2 ** i * bases))
        X = torch.concat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)
        # X *= self.displacements.unsqueeze(1)
        return bases + X

        # D = bases
        # for i in range(self.L + 1):
        #     D += X[i]
        #
        # # print(D.shape)
        # return D

    # def forward(self, bases, encodings, complex_encodings):
    #     X = [ ]
    #     # for i in range(self.Lpoint + 1):
    #     #     X.append(torch.sin(2 ** i * encodings))
    #     for i in range(self.Lcomplex + 1):
    #         X.append(torch.sin(2 ** i * complex_encodings))
    #     X.append(bases)
    #     X = torch.concat(X, dim=-1)
    #
    #     for i, linear in enumerate(self.encoding_linears):
    #         W, b = linear.weight, linear.bias
    #
    #         X = torch.matmul(X, W.t()) + b
    #         if i < len(self.encoding_linears) - 1:
    #             X = F.elu(X)
    #
    #     return bases + X
