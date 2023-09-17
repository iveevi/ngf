import matplotlib.pyplot as plt
import numpy as np
import os
import polyscope as ps
import seaborn as sns
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import trimesh
import time

from collections import namedtuple
from torch.utils.cpp_extension import load

from models import *

if len(sys.argv) < 3:
    print('Usage: python3 train.py <quad mesh> <reference> <directory>')
    # TODO: one homogenized python script...
    sys.exit(1)

def read_quad_mesh(path):
    quads, vertices = [], []
    with open(path, 'rb') as f:
        n_quads, n_vertices = np.fromfile(f, dtype=np.int32, count=2)
        quads = np.fromfile(f, dtype=np.int32, count=4 * n_quads).reshape(-1, 4)
        vertices = np.fromfile(f, dtype=np.float32, count=3 * n_vertices).reshape(-1, 3)

    return namedtuple('QuadMesh', ['quads', 'vertices'])(quads, vertices)

qm = read_quad_mesh(sys.argv[1])
ref = trimesh.load_mesh(sys.argv[2])

complexes = torch.from_numpy(qm.quads).cuda()
points = torch.from_numpy(qm.vertices).cuda()

encodings = torch.zeros((points.shape[0], POINT_ENCODING_SIZE), requires_grad=True, device='cuda')
complex_encodings = torch.zeros((complexes.shape[0], COMPLEX_ENCODING_SIZE), requires_grad=True, device='cuda')

# Get vertex -> complexes mapping
mapping = dict()
for i, c in enumerate(complexes):
    for v in c:
        mapping.setdefault(v.item(), []).append(i)

# print('mapping', len(mapping), mapping)

# Create a sparse matrix to convert complex encodings to vertex encodings
rows, cols, data = [], [], []
for v, cs in mapping.items():
    for c in cs:
        rows.append(v)
        cols.append(c)
        data.append(1.0/len(cs))

inds = (rows, cols)
matrix = torch.sparse_coo_tensor(inds, data, (points.shape[0], complexes.shape[0]), device='cuda')
# print('matrix', matrix.shape, matrix)

# vertex_encodings = matrix.mm(complex_encodings)
# print('vertex_encodings', vertex_encodings.shape)

print('Complexes:', complexes.shape)
print('Points:   ', points.shape)
print('Encodings:', encodings.shape)

def lerp(X, U, V):
    lp00 = X[:, 0, :].unsqueeze(1) * U.unsqueeze(-1) * V.unsqueeze(-1)
    lp01 = X[:, 1, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * V.unsqueeze(-1)
    lp10 = X[:, 3, :].unsqueeze(1) * U.unsqueeze(-1) * (1.0 - V.unsqueeze(-1))
    lp11 = X[:, 2, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * (1.0 - V.unsqueeze(-1))
    return lp00 + lp01 + lp10 + lp11

nsc = NSubComplex().cuda()

def chamfer(X, Y):
    d = torch.cdist(X, Y)
    return torch.min(d, dim=1)[0].mean() + torch.min(d, dim=0)[0].mean()

ref_vertices = torch.from_numpy(ref.vertices.astype(np.float32)).cuda()
ref_triangles = torch.from_numpy(ref.faces.astype(np.int32)).cuda()

def average_edge_length(V, T):
    v0 = V[T[:, 0], :]
    v1 = V[T[:, 1], :]
    v2 = V[T[:, 2], :]

    v01 = v1 - v0
    v02 = v2 - v0
    v12 = v2 - v1

    l01 = torch.norm(v01, dim=1)
    l02 = torch.norm(v02, dim=1)
    l12 = torch.norm(v12, dim=1)
    return (l01 + l02 + l12).mean()/3.0

def unormed_tri_normals(V, T):
    v0 = V[T[:, 0], :]
    v1 = V[T[:, 1], :]
    v2 = V[T[:, 2], :]
    v01 = v1 - v0
    v02 = v2 - v0
    return torch.cross(v01, v02, dim=1)

def tri_normals(V, T):
    v0 = V[T[:, 0], :]
    v1 = V[T[:, 1], :]
    v2 = V[T[:, 2], :]
    v01 = v1 - v0
    v02 = v2 - v0
    v12 = v2 - v1
    N0 = torch.cross(v01, v02, dim=1)
    N1 = torch.cross(v02, v12, dim=1)
    N = F.normalize(N0 + N1, dim=1)
    return N

def quad_normals(V, Q):
    v0 = V[Q[:, 0], :]
    v1 = V[Q[:, 1], :]
    v2 = V[Q[:, 2], :]
    v3 = V[Q[:, 3], :]

    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0

    N0 = torch.cross(v01, v02, dim=1)
    N1 = torch.cross(v02, v03, dim=1)

    # TODO: swap normals by doing a closest pass before starting
    # (and then using the sign)
    N = F.normalize(N0 + N1, dim=1)
    return -N

def tri_areas(V, T):
    v0 = V[T[:, 0], :]
    v1 = V[T[:, 1], :]
    v2 = V[T[:, 2], :]
    v01 = v1 - v0
    v02 = v2 - v0
    return torch.norm(torch.cross(v01, v02, dim=1), dim=1)/2.0

def quad_areas(V, Q):
    v0 = V[Q[:, 0], :]
    v1 = V[Q[:, 1], :]
    v2 = V[Q[:, 2], :]
    v3 = V[Q[:, 3], :]

    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0

    N0 = torch.cross(v01, v02, dim=1)
    N1 = torch.cross(v02, v03, dim=1)

    A0 = torch.norm(N0, dim=1)/2.0
    A1 = torch.norm(N1, dim=1)/2.0

    return A0 + A1

ref_normals = unormed_tri_normals(ref_vertices, ref_triangles)
nans = torch.isnan(ref_normals).any(dim=1)
print('NaNs:', nans.sum().item())
infs = torch.isinf(ref_normals).any(dim=1)
print('Infs:', infs.sum().item())

norms = torch.norm(ref_normals, dim=1)
print('Norms:', norms.min().item(), norms.max().item())

print('Loading geometry library...')
geom_cpp = load(name="geom_cpp",
        sources=[ "ext/geometry.cpp" ],
        extra_include_paths=[ "glm" ],
        build_directory="build",
)

print('Loading query library...')
query_cpp = load(name="query_cpp",
        sources=[ "ext/query.cu" ],
        extra_include_paths=[ "glm" ],
        build_directory="build",
)

print('Loading sample library...')
sample_cpp = load(name="sample_cpp",
        sources=[ "ext/sample.cu" ],
        extra_include_paths=[ "glm" ],
        build_directory="build",
)

def sample(sample_rate):
    C = matrix.mm(complex_encodings)
    U = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    V = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    U, V = torch.meshgrid(U, V)

    corner_points = points[complexes, :] # TODO: this should also be differentiable... (chamfer only?)
    corner_encodings = encodings[complexes, :]
    vertex_complex_encodings = C[complexes, :]

    U, V = U.reshape(-1), V.reshape(-1)
    U = U.repeat((complexes.shape[0], 1))
    V = V.repeat((complexes.shape[0], 1))

    lerped_points = lerp(corner_points, U, V).reshape(-1, 3)
    lerped_encodings = lerp(corner_encodings, U, V).reshape(-1, POINT_ENCODING_SIZE)
    lerped_complex_encodings = lerp(vertex_complex_encodings, U, V).reshape(-1, COMPLEX_ENCODING_SIZE)

    return lerped_points, lerped_encodings, lerped_complex_encodings

def make_cmap(complexes, LP):
    Cs = complexes.cpu().numpy()
    lp = LP.detach().cpu().numpy()

    cmap = dict()
    for i in range(Cs.shape[0]):
        for j in Cs[i]:
            if cmap.get(j) is None:
                cmap[j] = set()

        corners = np.array([
            0, sample_rate - 1,
            sample_rate * (sample_rate - 1),
            sample_rate ** 2 - 1
        ]) + (i * sample_rate ** 2)

        qvs = qm.vertices[Cs[i]]
        cvs = lp[corners]

        for j in range(4):
            # Find the closest corner
            dists = np.linalg.norm(qvs[j] - cvs, axis=1)
            closest = np.argmin(dists)
            cmap[Cs[i][j]].add(corners[closest])

    return cmap

optimizer = torch.optim.Adam(list(nsc.parameters()) + [ encodings, complex_encodings ], lr=1e-4)

history = {}
saves = []

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_normal_consistency, point_mesh_face_distance
from pytorch3d.ops import sample_points_from_meshes

from torchviz import make_dot

mref = Meshes(verts=[ ref_vertices ], faces=[ ref_triangles ])
print('mref', mref.verts_padded().shape, mref.faces_padded().shape)

# TODO: config file...

for sample_rate in [ 2, 4, 8, 16 ]:
    qindices = []
    for i in range(complexes.shape[0]):
        qi = quad_indices(sample_rate)
        qi += i * sample_rate ** 2
        qindices.append(qi)
    qindices = np.concatenate(qindices, axis=0)
    qindices_tensor = torch.from_numpy(qindices).cuda()

    tri_indices = []
    for i in range(complexes.shape[0]):
        ind = indices(sample_rate)
        ind += i * sample_rate ** 2
        tri_indices.append(ind)

    tri_indices = np.concatenate(tri_indices, axis=0)
    tri_indices_tensor = torch.from_numpy(tri_indices).cuda()

    # TODO: jittered sampling

    LP, LE, LC = sample(sample_rate)
    X = nsc(LP, LE, LC)

    cmap = make_cmap(complexes, LP)
    vertex_count = complexes.shape[0] * sample_rate ** 2
    I, remap = geom_cpp.sdc_weld(complexes.cpu(), cmap, vertex_count, sample_rate)
    I = I.cuda()
    
    aell = average_edge_length(X, I).detach()
    print('Average edge length:', aell.item())

    for i in tqdm.tqdm(range(10000)):
        LP, LE, LC = sample(sample_rate)
        X = nsc(LP, LE, LC)

        mopt = Meshes(verts=[ X ], faces=[ tri_indices_tensor ])
        mlap = Meshes(verts=[ X ], faces=[ I ])

        vertex_loss, normal_loss = chamfer_distance(
            x=mref.verts_padded(),
            y=mopt.verts_padded(),
            x_normals=mref.verts_normals_padded(),
            y_normals=mopt.verts_normals_padded(),
            abs_cosine=False)

        # sample_ref = sample_points_from_meshes(mref, 10000)[0]
        # sample_ref = Pointclouds(points=[ sample_ref ])
        # sample_ref_loss = point_mesh_face_distance(mopt, sample_ref)

        # sample_opt = sample_points_from_meshes(mopt, 10000)[0]
        # sample_opt = Pointclouds(points=[ sample_opt ])
        # sample_opt_loss = point_mesh_face_distance(mref, sample_opt)
        #
        # sample_loss = sample_ref_loss + sample_opt_loss

        areas = tri_areas(X, tri_indices_tensor)
        area_loss = torch.mean(areas) + torch.var(areas)

        laplacian_loss = mesh_laplacian_smoothing(mlap, method='uniform')
        consistency_loss = mesh_normal_consistency(mlap) #TODO: use with mlap

        # loss = vertex_loss + normal_loss + area_loss + laplacian_loss
        # loss = vertex_loss + sample_ref_loss + 1e-2 * normal_loss + 1e-2 * laplacian_loss
        loss = vertex_loss + aell * normal_loss + 1e-2 * laplacian_loss + aell * consistency_loss
        # loss = sample_ref_loss + 1e-3 * normal_loss # + 1e-3 * laplacian_loss
        dot = make_dot(loss, params=dict(nsc.named_parameters()))

        history.setdefault('vertex:loss', []).append(vertex_loss.item())
        history.setdefault('normal:loss', []).append(normal_loss.item())
        # history.setdefault('sample:ref sample loss', []).append(sample_ref_loss.item())
        # history.setdefault('sample:opt sample loss', []).append(sample_opt_loss.item())
        history.setdefault('miscellaneous:laplacian loss', []).append(laplacian_loss.item())
        history.setdefault('miscellaneous:consistency loss', []).append(consistency_loss.item())
        history.setdefault('miscellaneous:total loss', []).append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    saves.append((X.detach().cpu().numpy(), qindices))

# print('Final displacement factors:', nsc.displacements)
# dot.render('loss_graph', format='png')

# Plot the loss
groups = {}
for k, v in history.items():
    origin = k.split(':')[0]
    label = k.split(':')[1]
    groups.setdefault(origin, dict())[label] = v

sns.set()

fig, axs = plt.subplots(1, len(groups), figsize=(10 * len(groups), 10))
if len(groups) == 1:
    axs = [axs]

for ax, origin, group in zip(axs, groups.keys(), groups.values()):
    for label, values in group.items():
        ax.plot(values, label=label)

    ax.legend()
    ax.set_title(origin)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_yscale('log')

fig.tight_layout()
plt.savefig('loss.png')

# Display results
ps.init()

q = ps.register_surface_mesh("quad", qm.vertices, qm.quads)
r = ps.register_surface_mesh("ref", ref.vertices, ref.faces)
r.add_color_quantity("color", (0.5 * ref_normals + 0.5).detach().cpu().numpy(), defined_on='faces')

for i, (X, Q) in enumerate(saves):
    ps.register_surface_mesh("save" + str(i), X, Q)

ps.show()
