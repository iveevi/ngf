import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POINT_ENCODING_SIZE = 10

# def indices(N):
#     gim_indices = []
#     for i in range(N - 1):
#         for j in range(N - 1):
#             gim_indices.append([i * N + j, (i + 1) * N + j, i * N + j + 1])
#             gim_indices.append([(i + 1) * N + j, (i + 1) * N + j + 1, i * N + j + 1])
#     return np.array(gim_indices).reshape(-1, 3).astype(np.int32)
#
# def quad_indices(N):
#     gim_indices = []
#     for i in range(N - 1):
#         for j in range(N - 1):
#             gim_indices.append([i * N + j, i * N + j + 1, (i + 1) * N + j + 1, (i + 1) * N + j])
#     return np.array(gim_indices).reshape(-1, 4).astype(np.int32)

class NSubComplex(nn.Module):
    L = 8

    def __init__(self) -> None:
        super(NSubComplex , self).__init__()

        # TODO: try reducing the model to 64 hidden neurons (and maybe 2 layers rather than one)
        # Pass configuration as constructor arguments
        self.encoding_linears = [
            nn.Linear(POINT_ENCODING_SIZE + (2 * self.L + 1) * 3, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 3)
        ]

        self.encoding_linears = nn.ModuleList(self.encoding_linears)

    def forward(self, bases, encodings):
        X = [ encodings, bases ]
        for i in range(self.L):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]
        X = torch.concat(X, dim=-1)

        for i, linear in enumerate(self.encoding_linears):
            X = linear(X)
            if i < len(self.encoding_linears) - 1:
                X = torch.sin(X)

        return bases + X
    
    def serialize(self, complexes, corner_points, corner_encodings, path):
        l0, l1, l2 = self.encoding_linears

        with open(path, 'wb') as f:
            f.write(np.array([
                complexes.shape[0],
                corner_points.shape[0],
                POINT_ENCODING_SIZE,
            ], dtype=np.int32).tobytes())
            
            # Complexes
            complexes_bytes = complexes.cpu().numpy().astype(np.int32).tobytes()
            print('complexes_bytes:', len(complexes_bytes))
            f.write(complexes_bytes)

            # Corner points and encodings
            K = corner_points.shape[0]
            assert(corner_points.shape[0] == K)
            assert(corner_encodings.shape[0] == K)
            corner_points_bytes = corner_points.detach().cpu().numpy().astype(np.float32).tobytes()
            corner_encodings_bytes = corner_encodings.detach().cpu().numpy().astype(np.float32).tobytes()
            print('corner_points_bytes:', len(corner_points_bytes))
            print('corner_encodings_bytes:', len(corner_encodings_bytes))
            f.write(corner_points_bytes)
            f.write(corner_encodings_bytes)

            # Write the model parameters
            W0, H0 = l0.weight.shape
            print('W0, H0:', W0, H0)
            l0_weights = l0.weight.detach().cpu().numpy().astype(np.float32).tobytes()
            l0_bias = l0.bias.detach().cpu().numpy().astype(np.float32).tobytes()
            f.write(np.array([W0, H0], dtype=np.int32).tobytes())
            bytes = f.write(l0_weights)
            print('l0_weights:', bytes)
            bytes = f.write(l0_bias)
            print('l0_bias:', bytes)
            
            W1, H1 = l1.weight.shape
            print('W1, H1:', W1, H1)
            l1_weights = l1.weight.detach().cpu().numpy().astype(np.float32).tobytes()
            l1_bias = l1.bias.detach().cpu().numpy().astype(np.float32).tobytes()
            f.write(np.array([W1, H1], dtype=np.int32).tobytes())
            bytes = f.write(l1_weights)
            print('l1_weights:', bytes)
            bytes = f.write(l1_bias)
            print('l1_bias:', bytes)

            W2, H2 = l2.weight.shape
            print('W2, H2:', W2, H2)
            l2_weights = l2.weight.detach().cpu().numpy().astype(np.float32).tobytes()
            l2_bias = l2.bias.detach().cpu().numpy().astype(np.float32).tobytes()
            f.write(np.array([W2, H2], dtype=np.int32).tobytes())
            bytes = f.write(l2_weights)
            print('l2_weights:', bytes)
            bytes = f.write(l2_bias)
            print('l2_bias:', bytes)
