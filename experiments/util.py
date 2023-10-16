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

def compute_face_normals(verts, faces):
    """
    Compute per-face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)

    v = [verts.index_select(1, fi[0]),
                 verts.index_select(1, fi[1]),
                 verts.index_select(1, fi[2])]

    c = torch.cross(v[1] - v[0], v[2] - v[0])
    n = c / torch.norm(c, dim=0)
    return n

def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))

def compute_vertex_normals(verts, faces, face_normals):
    """
    Compute per-vertex normals from face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    face_normals : torch.Tensor
        Per-face normals
    """

    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)
    normals = torch.zeros_like(verts)

    v = [verts.index_select(1, fi[0]),
             verts.index_select(1, fi[1]),
             verts.index_select(1, fi[2])]

    for i in range(3):
        d0 = v[(i + 1) % 3] - v[i]
        d0 = d0 / torch.norm(d0)
        d1 = v[(i + 2) % 3] - v[i]
        d1 = d1 / torch.norm(d1)
        d = torch.sum(d0 * d1, dim=0)
        # print('# of zeros: ', (d == 0).sum())
        # print('# of gones: ', (d >= 1).sum())
        # print('min max: ', d.min(), d.max())
        face_angle = safe_acos(d)
        assert not torch.isnan(face_angle).any()
        nn =  face_normals * face_angle
        for j in range(3):
            normals[j].index_add_(0, fi[i], nn[j])

    return normals.transpose(0, 1)
    # return (normals / torch.norm(normals, dim=0)).transpose(0, 1)
