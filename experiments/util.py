import numpy as np
import torch

from mlp import *

def lerp(X, U, V):
    lp00 = X[:, 0, :].unsqueeze(1) * U.unsqueeze(-1) * V.unsqueeze(-1)
    lp01 = X[:, 1, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * V.unsqueeze(-1)
    lp10 = X[:, 3, :].unsqueeze(1) * U.unsqueeze(-1) * (1.0 - V.unsqueeze(-1))
    lp11 = X[:, 2, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * (1.0 - V.unsqueeze(-1))
    return lp00 + lp01 + lp10 + lp11

def sample(complexes, corners, encodings, sample_rate, kernel=lerp):
    U = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    V = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    U, V = torch.meshgrid(U, V, indexing='ij')

    corner_points = corners[complexes, :]
    corner_encodings = encodings[complexes, :]

    U, V = U.reshape(-1), V.reshape(-1)
    U = U.repeat((complexes.shape[0], 1))
    V = V.repeat((complexes.shape[0], 1))

    lerped_points = lerp(corner_points, U, V).reshape(-1, 3)
    lerped_encodings = kernel(corner_encodings, U, V).reshape(-1, POINT_ENCODING_SIZE)

    return lerped_points, lerped_encodings, torch.stack([U, V], dim=-1).squeeze(0)

def indices(C, sample_rate=16):
    triangles = []
    for c in range(C.shape[0]):
        offset = c * sample_rate * sample_rate
        for i in range(sample_rate - 1):
            for j in range(sample_rate - 1):
                a = offset + i * sample_rate + j
                c = offset + (i + 1) * sample_rate + j
                b, d = a + 1, c + 1
                triangles.append([a, b, c])
                triangles.append([b, d, c])

    return np.array(triangles)

def shorted_indices(V, C, sample_rate=16):
    triangles = []
    for c in range(C.shape[0]):
        offset = c * sample_rate * sample_rate
        for i in range(sample_rate - 1):
            for j in range(sample_rate - 1):
                a = offset + i * sample_rate + j
                c = offset + (i + 1) * sample_rate + j
                b, d = a + 1, c + 1

                vs = V[[a, b, c, d]]
                d0 = np.linalg.norm(vs[0] - vs[3])
                d1 = np.linalg.norm(vs[1] - vs[2])

                if d0 < d1:
                    triangles.append([a, d, b])
                    triangles.append([a, c, d])
                else:
                    triangles.append([a, c, b])
                    triangles.append([b, c, d])

    return np.array(triangles)

def color_code_complexes(C, sample_rate=16):
    color_wheel = [
            np.array([0.700, 0.300, 0.300]),
            np.array([0.700, 0.450, 0.300]),
            np.array([0.700, 0.600, 0.300]),
            np.array([0.650, 0.700, 0.300]),
            np.array([0.500, 0.700, 0.300]),
            np.array([0.350, 0.700, 0.300]),
            np.array([0.300, 0.700, 0.400]),
            np.array([0.300, 0.700, 0.550]),
            np.array([0.300, 0.700, 0.700]),
            np.array([0.300, 0.550, 0.700]),
            np.array([0.300, 0.400, 0.700]),
            np.array([0.350, 0.300, 0.700]),
            np.array([0.500, 0.300, 0.700]),
            np.array([0.650, 0.300, 0.700]),
            np.array([0.700, 0.300, 0.600]),
            np.array([0.700, 0.300, 0.450])
    ]

    complex_face_colors = []
    for i in range(C.shape[0]):
        color = color_wheel[i % len(color_wheel)]
        complex_face_colors.append(np.tile(color, (2 * (sample_rate - 1) ** 2, 1)))

    return np.concatenate(complex_face_colors)