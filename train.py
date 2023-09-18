import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import polyscope as ps
import seaborn as sns
import shutil
import torch
import torch.nn.functional as F
import tqdm
import trimesh

from collections import namedtuple
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_normal_consistency, point_mesh_face_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from torch.utils.cpp_extension import load
from torchviz import make_dot

from models import *

parser = argparse.ArgumentParser(description='Configure the experiment.')
parser.add_argument('-q', '--quads', type=str, help='Path to the base quad mesh.', required=True)
parser.add_argument('-r', '--reference', type=str, help='Path to the reference mesh.', required=True)
parser.add_argument('-d', '--directory', type=str, help='Path to the result directory.', required=True)
args = parser.parse_args()

def read_quad_mesh(path):
    quads, vertices = [], []
    with open(path, 'rb') as f:
        n_quads, n_vertices = np.fromfile(f, dtype=np.int32, count=2)
        quads = np.fromfile(f, dtype=np.int32, count=4 * n_quads).reshape(-1, 4)
        vertices = np.fromfile(f, dtype=np.float32, count=3 * n_vertices).reshape(-1, 3)

    return namedtuple('QuadMesh', ['quads', 'vertices'])(quads, vertices)

qm = read_quad_mesh(args.quads)
ref = trimesh.load_mesh(args.reference)
name = os.path.basename(args.reference).split('.')[0]

complexes = torch.from_numpy(qm.quads).cuda()
points = torch.from_numpy(qm.vertices).cuda()

encodings = torch.zeros((points.shape[0], POINT_ENCODING_SIZE), requires_grad=True, device='cuda')

# Compute normal vectors for each complex
complex_normals = torch.zeros((complexes.shape[0], 3), device='cuda')
for i, c in enumerate(complexes):
    v = points[c]
    n0 = torch.cross(v[1] - v[0], v[2] - v[0])
    n1 = torch.cross(v[2] - v[0], v[3] - v[0])
    complex_normals[i] = n0 + n1

# Get vertex -> complexes mapping
mapping = dict()
for i, c in enumerate(complexes):
    for v in c:
        mapping.setdefault(v.item(), []).append(i)

# Compute per-vertex encodings
normals  = torch.zeros((points.shape[0], 3), device='cuda')
for v, cs in mapping.items():
    n = torch.stack([complex_normals[c] for c in cs]).sum(dim=0)
    normals[v] = F.normalize(n, dim=0)

# Convert to spherical coordinates and normalize
phi = 0.5 * (torch.atan2(normals[:, 1], normals[:, 0])/np.pi) + 0.5
theta = torch.acos(normals[:, 2])/np.pi
normals = torch.stack([phi, theta], dim=1)

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

def sample(sample_rate):
    U = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    V = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    U, V = torch.meshgrid(U, V)

    corner_points = points[complexes, :] # TODO: this should also be differentiable... (chamfer only?)
    corner_normals = normals[complexes, :]
    corner_encodings = encodings[complexes, :]

    U, V = U.reshape(-1), V.reshape(-1)
    U = U.repeat((complexes.shape[0], 1))
    V = V.repeat((complexes.shape[0], 1))

    lerped_points = lerp(corner_points, U, V).reshape(-1, 3)
    lerped_normals = lerp(corner_normals, U, V).reshape(-1, 2)
    lerped_encodings = lerp(corner_encodings, U, V).reshape(-1, POINT_ENCODING_SIZE)

    return lerped_points, lerped_normals, lerped_encodings

# def make_cmap(complexes, LP):
#     Cs = complexes.cpu().numpy()
#     lp = LP.detach().cpu().numpy()
#
#     cmap = dict()
#     for i in range(Cs.shape[0]):
#         for j in Cs[i]:
#             if cmap.get(j) is None:
#                 cmap[j] = set()
#
#         corners = np.array([
#             0, sample_rate - 1,
#             sample_rate * (sample_rate - 1),
#             sample_rate ** 2 - 1
#         ]) + (i * sample_rate ** 2)
#
#         qvs = qm.vertices[Cs[i]]
#         cvs = lp[corners]
#
#         for j in range(4):
#             # Find the closest corner
#             dists = np.linalg.norm(qvs[j] - cvs, axis=1)
#             closest = np.argmin(dists)
#             cmap[Cs[i][j]].add(corners[closest])
#
#     return cmap

optimizer = torch.optim.Adam(list(nsc.parameters()) + [ encodings, points ], lr=1e-3)

history = {}
saves = []

ref_vertices = torch.from_numpy(ref.vertices.astype(np.float32)).cuda()
ref_triangles = torch.from_numpy(ref.faces.astype(np.int32)).cuda()
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

    LP, LN, LE = sample(sample_rate)
    X = nsc(LP, LN, LE)

    aell = average_edge_length(X, tri_indices_tensor).detach()
    for i in tqdm.tqdm(range(sample_rate * 1000), desc=f'Optimizing {name} at {sample_rate}x{sample_rate}'):
        LP, LN, LE = sample(sample_rate)
        X = nsc(LP, LN, LE)

        mopt = Meshes(verts=[ X ], faces=[ tri_indices_tensor ])

        vertex_loss, normal_loss = chamfer_distance(
            x=mref.verts_padded(),
            y=mopt.verts_padded(),
            x_normals=mref.verts_normals_padded(),
            y_normals=mopt.verts_normals_padded(),
            abs_cosine=False)

        laplacian_loss = mesh_laplacian_smoothing(mopt, method='uniform')
        consistency_loss = mesh_normal_consistency(mopt) #TODO: use with mlap

        loss = vertex_loss + aell * normal_loss + aell * consistency_loss/(sample_rate ** 2)
        if sample_rate <= 4:
            loss += aell * laplacian_loss

        dot = make_dot(loss, params=dict(nsc.named_parameters()))

        history.setdefault('vertex:loss', []).append(vertex_loss.item())
        history.setdefault('normal:loss', []).append(normal_loss.item())
        history.setdefault('miscellaneous:laplacian loss', []).append(laplacian_loss.item())
        history.setdefault('miscellaneous:consistency loss', []).append(consistency_loss.item())
        history.setdefault('miscellaneous:total loss', []).append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    saves.append((X.detach().cpu().numpy(), qindices))

# Last iteration
optimizer = torch.optim.Adam([ encodings ], lr=1e-4)

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

LP, LN, LE = sample(sample_rate)
X = nsc(LP, LN, LE)

aell = average_edge_length(X, tri_indices_tensor).detach()
for i in tqdm.tqdm(range(sample_rate * 1000), desc=f'Optimizing {name} at {sample_rate}x{sample_rate}'):
    LP, LN, LE = sample(sample_rate)
    X = nsc(LP, LN, LE)

    mopt = Meshes(verts=[ X ], faces=[ tri_indices_tensor ])

    vertex_loss, normal_loss = chamfer_distance(
        x=mref.verts_padded(),
        y=mopt.verts_padded(),
        x_normals=mref.verts_normals_padded(),
        y_normals=mopt.verts_normals_padded(),
        abs_cosine=False)

    laplacian_loss = mesh_laplacian_smoothing(mopt, method='uniform')
    consistency_loss = mesh_normal_consistency(mopt) #TODO: use with mlap

    loss = vertex_loss + aell * normal_loss
    if sample_rate <= 4:
        loss += aell * laplacian_loss

    dot = make_dot(loss, params=dict(nsc.named_parameters()))

    history.setdefault('vertex:loss', []).append(vertex_loss.item())
    history.setdefault('normal:loss', []).append(normal_loss.item())
    history.setdefault('miscellaneous:laplacian loss', []).append(laplacian_loss.item())
    history.setdefault('miscellaneous:consistency loss', []).append(consistency_loss.item())
    history.setdefault('miscellaneous:total loss', []).append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

saves.append((X.detach().cpu().numpy(), qindices))

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

# Save results
output_dir = args.directory
model_file = os.path.join(output_dir, 'model.bin')
complexes_file = os.path.join(output_dir, 'complexes.bin')
corner_points_file = os.path.join(output_dir, 'points.bin')
corner_encodings_file = os.path.join(output_dir, 'encodings.bin')

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir)
torch.save(nsc, model_file)
torch.save(complexes, complexes_file)
torch.save(points, corner_points_file)
torch.save(encodings, corner_encodings_file)
plt.savefig(os.path.join(output_dir, 'loss.png'))

filename = args.reference
extension = os.path.splitext(filename)[1]
print('Copying', filename, 'to', os.path.join(output_dir, 'ref' + extension))
shutil.copyfile(filename, os.path.join(output_dir, 'ref' + extension))

# Display results
ps.init()

r = ps.register_surface_mesh("ref", ref.vertices, ref.faces)
for i, (X, Q) in enumerate(saves):
    ps.register_surface_mesh("save" + str(i), X, Q)

# I = np.arange(sample_rate * sample_rate)
# I = np.tile(I, (attentive_indices.shape[0], 1)).reshape(-1)
# O = attentive_indices.repeat(sample_rate * sample_rate) * (sample_rate * sample_rate) + I
#
# ps.register_surface_mesh("attentive", X[O], att_tri_indices)

ps.show()
