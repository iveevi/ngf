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

if len(sys.argv) < 4:
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

#FIXME:
encodings = torch.zeros((points.shape[0], ENCODING_SIZE), requires_grad=True, device='cuda')
# encodings = torch.randn((points.shape[0], ENCODING_SIZE), requires_grad=True, device='cuda')

print('Complexes:', complexes.shape)
print('Points:   ', points.shape)
print('Encodings:', encodings.shape)

# Uniform UV coordinates for each complex
# sample_rate = 2

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

def quad_area(V, Q):
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

    return torch.concatenate((A0, A1), dim=0)

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
    U = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    V = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    U, V = torch.meshgrid(U, V)

    corner_points = points[complexes, :] # TODO: this should also be differentiable... (chamfer only?)
    corner_encodings = encodings[complexes, :]

    U, V = U.reshape(-1), V.reshape(-1)
    U = U.repeat((complexes.shape[0], 1))
    V = V.repeat((complexes.shape[0], 1))

    lerped_points = lerp(corner_points, U, V).reshape(-1, 3)
    lerped_encodings = lerp(corner_encodings, U, V).reshape(-1, ENCODING_SIZE)
    return lerped_points, lerped_encodings

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

def diffusion_matrix(T, Nv, factor = 20):
    # Assume T is a numpy matrix (N, 3) of triangle indices
    data, rows, columns = [], [], []

    vgraph = {}
    for i in range(T.shape[0]):
        v0, v1, v2 = T[i]

        vgraph.setdefault(v0, set()).add(v1)
        vgraph.setdefault(v0, set()).add(v2)
        vgraph.setdefault(v1, set()).add(v0)
        vgraph.setdefault(v1, set()).add(v2)
        vgraph.setdefault(v2, set()).add(v0)
        vgraph.setdefault(v2, set()).add(v1)

    for v, adj in vgraph.items():
        for a in adj:
            data.append(1)
            rows.append(v)
            columns.append(a)

        data.append(-len(adj))
        rows.append(v)
        columns.append(v)

    data = cupy.array(data, dtype=cupy.float32)
    rows = cupy.array(rows, dtype=cupy.int32)
    columns = cupy.array(columns, dtype=cupy.int32)

    L = coo_matrix((data, (rows, columns)), shape=(Nv, Nv))

    I_ones = cupy.ones(Nv, dtype=cupy.float32)
    I_rows = cupy.arange(Nv, dtype=cupy.int32)
    I = coo_matrix((I_ones, (I_rows, I_rows)))

    return I - factor * L

optimizer = torch.optim.Adam(list(nsc.parameters()) + [ encodings ], lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

history = {}
saves = []

for sample_rate in [ 2, 4 ]:
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

    # Prepare an accelerator
    LP, LE = sample(sample_rate)
    X = nsc(LP, LE)

    acc = query_cpp.accelerator(X, ref_vertices, ref_normals, ref_triangles, sample_rate, 1.5)
    print('acc:', acc)

    for i in tqdm.tqdm(range(1000)):
        LP, LE = sample(sample_rate)
        X = nsc(LP, LE)

        optimizer.zero_grad()

        # y, n = acc.closest(X, sample_rate)
        # print('y:', y.shape, 'n:', n.shape)

        # Y, N = query_cpp.closest(X, ref_vertices, ref_normals, ref_triangles)
        Y, N = acc.closest(X, sample_rate)
        vertex_loss = torch.mean(torch.pow(X - Y, 2))

        t = np.sin(time.perf_counter())
        Xs, Ns = sample_cpp.sample(ref_vertices, ref_normals, ref_triangles, 1000, t)
        T, B = query_cpp.closest_bary(Xs, X, tri_indices_tensor)
        Tris = tri_indices_tensor[T]
        V0, V1, V2 = X[Tris[:, 0], :], X[Tris[:, 1], :], X[Tris[:, 2], :]
        Vh = V0 * B[:, 0].unsqueeze(1) + V1 * B[:, 1].unsqueeze(1) + V2 * B[:, 2].unsqueeze(1)
        sampled_vertex_loss = torch.mean(torch.pow(Xs - Vh, 2))

        history.setdefault('opt:vertex loss', []).append(vertex_loss.item())
        history.setdefault('opt:sampled vertex loss', []).append(sampled_vertex_loss.item())
        history.setdefault('opt:lr', []).append(scheduler.get_last_lr()[0])

        loss = vertex_loss + sampled_vertex_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    saves.append((X.detach().cpu().numpy(), qindices_tensor.cpu().numpy(), tri_indices_tensor.cpu().numpy(), sample_rate))

structs = []

for sample_rate in [ 8 ]:
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

    nsc_ids = np.arange(complexes.shape[0]).astype(np.int32)
    nsc_ids = np.tile(nsc_ids, ((sample_rate - 1) ** 2, 1)).T.reshape(-1)
    nsc_ids_tensor = torch.from_numpy(nsc_ids).cuda()

    for i in range(2):
        LP, LE = sample(sample_rate)
        X = nsc(LP, LE)

        V = X.detach().clone().requires_grad_(True)
        vopt = torch.optim.SGD([ V ], lr=1e-2)

        cmap = make_cmap(complexes, LP)
        I, remap = geom_cpp.sdc_weld(complexes.cpu(), X.cpu(), cmap, sample_rate)
        I = I.cuda()

        acc = query_cpp.accelerator(X, ref_vertices, ref_normals, ref_triangles, sample_rate, 1.5)
        for i in tqdm.tqdm(range(1000)):
            vopt.zero_grad()

            # Compute triangle normals
            Tn = tri_normals(V, I)

            # Find closest points on the reference
            # Y, N = query_cpp.closest(V, ref_vertices, ref_normals, ref_triangles)
            Y, N = acc.closest(X, sample_rate)
            vertex_loss = torch.sum(torch.pow(V - Y, 2))

            # Sampling points on the ref
            t = np.sin(time.perf_counter())
            Xs, Ns = sample_cpp.sample(ref_vertices, ref_normals, ref_triangles, 1000, t)
            T, B = query_cpp.closest_bary(Xs, V, I)
            Tris = tri_indices_tensor[T]
            V0, V1, V2 = V[Tris[:, 0], :], V[Tris[:, 1], :], V[Tris[:, 2], :]
            Vh = V0 * B[:, 0].unsqueeze(1) + V1 * B[:, 1].unsqueeze(1) + V2 * B[:, 2].unsqueeze(1)
            sampled_vertex_loss = torch.sum(torch.pow(Xs - Vh, 2)) # * (V.shape[0] / Xs.shape[0])

            loss = vertex_loss + sampled_vertex_loss

            history.setdefault('vopt:loss', []).append(loss.item())
            history.setdefault('vopt:vertex_loss', []).append(vertex_loss.item())
            history.setdefault('vopt:sampled_vertex_loss', []).append(sampled_vertex_loss.item())

            loss.backward()
            vopt.step()

            # Laplacian smoothing every now and then
            if (i > 0 and i < 500) and i % 50 == 0:
                Vnew = geom_cpp.laplacian_smooth(V.detach().cpu(), I.cpu(), 0.9)
                with torch.no_grad():
                    V.data.copy_(Vnew.cuda())
                
                acc = query_cpp.accelerator(V, ref_vertices, ref_normals, ref_triangles, sample_rate, 1.5)

        V = geom_cpp.sdc_separate(V.detach().cpu(), remap).cuda()
        Vn = tri_normals(V, tri_indices_tensor)
        aell = average_edge_length(V, tri_indices_tensor)

        optimizer = torch.optim.Adam(list(nsc.parameters()) + [ encodings ], lr=1e-3)

        for i in tqdm.tqdm(range(10000)):
            LP, LE = sample(sample_rate)
            X = nsc(LP, LE)

            vertex_loss = torch.mean(torch.pow(X - V, 2))

            Xn = tri_normals(X, tri_indices_tensor)
            normal_loss = aell * torch.mean(torch.pow(Xn - Vn, 2))

            loss = vertex_loss + 10 * normal_loss

            history.setdefault('nsc:loss', []).append(loss.item())
            history.setdefault('nsc:vertex_loss', []).append(vertex_loss.item())
            history.setdefault('nsc:normal_loss', []).append(normal_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    structs.append((V.cpu().numpy(), qindices_tensor.cpu().numpy()))
    saves.append((X.detach().cpu().numpy(), qindices_tensor.cpu().numpy(), tri_indices_tensor.cpu().numpy(), sample_rate))

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
output_dir = sys.argv[3]
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

filename = sys.argv[2]
extension = os.path.splitext(filename)[1]
print('Copying', filename, 'to', os.path.join(output_dir, 'ref' + extension))
shutil.copyfile(filename, os.path.join(output_dir, 'ref' + extension))

# Display results
ps.init()

q = ps.register_surface_mesh("quad", qm.vertices, qm.quads)
r = ps.register_surface_mesh("ref", ref.vertices, ref.faces)
r.add_color_quantity("color", (0.5 * ref_normals + 0.5).detach().cpu().numpy(), defined_on='faces')

for i, (X, Q, T, S) in enumerate(saves):
    ps.register_surface_mesh("save" + str(i), X, Q)

    # Save to output directory as well
    m = trimesh.Trimesh(vertices=X, faces=Q)
    m.export(os.path.join(output_dir, f'mesh{S}x{S}.obj'))
    print('exported to', os.path.join(output_dir, f'mesh{S}x{S}.obj'))

for i, (V, Q) in enumerate(structs):
    ps.register_surface_mesh("struct" + str(i), V, Q)

ps.show()
