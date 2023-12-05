import matplotlib.pyplot as plt
import meshio
import numpy as np
import polyscope as ps
import sys
import torch

import optext

from tqdm import trange

# Icosahedron
def icosahedron():
    phi = (1 + 5 ** 0.5) / 2
    points = [
        [ -1,  phi,  0 ],
        [ 1,  phi,  0 ],
        [ -1, -phi,  0 ],
        [ 1, -phi,  0 ],
        [ 0, -1,  phi ],
        [ 0,  1,  phi ],
        [ 0, -1, -phi ],
        [ 0,  1, -phi ],
        [ phi,  0, -1 ],
        [ phi,  0,  1 ],
        [ -phi,  0, -1 ],
        [ -phi,  0,  1 ],
    ]

    points = np.array(points, dtype=np.float32)
    points /= np.linalg.norm(points, axis=-1, keepdims=True)

    faces = [
        [ 0, 11, 5 ],
        [ 0, 5, 1 ],
        [ 0, 1, 7 ],
        [ 0, 7, 10 ],
        [ 0, 10, 11 ],

        [ 1, 5, 9 ],
        [ 5, 11, 4 ],
        [ 11, 10, 2 ],
        [ 10, 7, 6 ],
        [ 7, 1, 8 ],

        [ 3, 9, 4 ],
        [ 3, 4, 2 ],
        [ 3, 2, 6 ],
        [ 3, 6, 8 ],
        [ 3, 8, 9 ],

        [ 4, 9, 5 ],
        [ 2, 4, 11 ],
        [ 6, 2, 10 ],
        [ 8, 6, 7 ],
        [ 9, 8, 1 ],
    ]

    return points, faces

def icorefine(points, faces):
    new_points = []
    new_faces  = []

    for face in faces:
        i = len(new_points)

        a = points[face[0]]
        b = points[face[1]]
        c = points[face[2]]

        ab = (a + b) / 2
        bc = (b + c) / 2
        ca = (c + a) / 2

        new_points.append(a)
        new_points.append(b)
        new_points.append(c)
        new_points.append(ab / np.linalg.norm(ab))
        new_points.append(bc / np.linalg.norm(bc))
        new_points.append(ca / np.linalg.norm(ca))

        new_faces.append([ i + 0, i + 3, i + 5 ])
        new_faces.append([ i + 3, i + 1, i + 4 ])
        new_faces.append([ i + 5, i + 4, i + 2 ])
        new_faces.append([ i + 3, i + 4, i + 5 ])

    return np.array(new_points, dtype=np.float32), np.array(new_faces, dtype=np.int32)

p, f = icosahedron()

for i in range(3):
    p, f = icorefine(p, f)

ps.init()
m = ps.register_surface_mesh("icosahedron", p, f)

# Compute phi and theta for each point
phi = np.arctan2(p[:, 2], p[:, 0])
theta = np.arccos(p[:, 1])

m.add_scalar_quantity("phi", phi)
m.add_scalar_quantity("theta", theta)

ps.show()

exit()

# Load the target face object
target = meshio.read('igea.obj')
target_points = torch.tensor(target.points, dtype=torch.float32, device='cuda')
target_faces  = torch.tensor(target.cells_dict['triangle'], dtype=torch.int32, device='cuda')

extent = target_points.max(dim=0)[0] - target_points.min(dim=0)[0]
radius = extent.max() / 2

# Make the acceleration structure for optimization
geometry = optext.geometry(target_points.cpu(), target_faces.cpu())
casdf = optext.cached_grid(geometry, 64)
print(casdf)
print(geometry)

# Generate points on a sphere
n_z     = 300
n_theta = 200

# TODO: Feature on the plane...
# Optimize to fit a face shape (with some gaussians and other parameteres)

# Number of gaussians on top of the surface
n_kernels = 100

def generator(theta, z, A):
    r = (1 - z * z).sqrt()

    # Modify the radius with gaussians
    for i in range(n_kernels):
        # thetap = (np.pi - theta).abs()
        thetap = torch.sin(theta)
        r += amps[i].abs() * radius * torch.exp(
            -((thetap - centers[i, 0]) / sigmas[i, 0]).pow(2) / 2
            -((z - centers[i, 1]) / sigmas[i, 1]).pow(2) / 2
        )

    xs = radius * r * theta.cos() * A[0]
    ys = radius * z * A[2]
    zs = radius * r * theta.sin() * A[1]
    return torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)

# Parameters
A = torch.tensor([ 1.0,  1.0,  1.0 ], dtype=torch.float32, requires_grad=True, device='cuda')

# Parameters for the gaussians
centers = torch.rand(n_kernels, 2, device='cuda') * 2 - 1
sigmas  = torch.rand(n_kernels, 2, device='cuda') * 0.1 + 0.1
amps    = torch.rand(n_kernels, device='cuda') * 0.1 + 0.1

centers.requires_grad = True
sigmas.requires_grad  = True
amps.requires_grad    = True

# Domain
theta    = torch.linspace(0, 2 * np.pi, n_theta, device='cuda')
z        = torch.linspace(-1, 1, n_z, device='cuda')
theta, z = torch.meshgrid(theta, z)

points = generator(theta, z, A)
print(points.shape)

# Generate index buffer
faces = []
for i in range(n_theta):
    for j in range(1, n_z - 2):
        ni = (i + 1) % n_theta
        nj = (j + 1) % n_z

        faces.append([
            i * n_z + j,
            i * n_z + nj,
            ni * n_z + nj
        ])

        faces.append([
            i * n_z + j,
            ni * n_z + nj,
            ni * n_z + j
        ])

# Add triangles at the end (z = 0 and z = n_z - 1)
for i in range(n_theta):
    ni = (i + 1) % n_theta

    faces.append([
        0,
        i * n_z + 1,
        ni * n_z + 1
    ])

    faces.append([
        n_theta * n_z - 1,
        ni * n_z + n_z - 2,
        i * n_z + n_z - 2,
    ])

faces = np.array(faces)
faces = torch.tensor(faces, dtype=torch.int32, device='cuda')

# Optimization
optimizer = torch.optim.Adam([ A, centers, sigmas, amps ], lr=0.1)

closest  = torch.zeros_like(points)
bary     = torch.zeros_like(points)
distance = torch.zeros(points.shape[0], dtype=torch.float32, device='cuda')
index    = torch.zeros(points.shape[0], dtype=torch.int32, device='cuda')

samples = 1000
samples_bary = torch.zeros(samples, 3, device='cuda')
samples_index = torch.zeros(samples, dtype=torch.int32, device='cuda')

def sample_surface(points, faces, count):
    indices = torch.randint(0, faces.shape[0], (count,), device='cuda')
    bary    = torch.rand(count, 2, device='cuda')
    bary    = torch.stack([bary[:, 0], 1 - bary.sum(dim=-1), bary[:, 1]], dim=-1)
    assert torch.all(bary.sum(dim=-1) < 1 + 1e-5)

    v0 = points[faces[indices, 0]]
    v1 = points[faces[indices, 1]]
    v2 = points[faces[indices, 2]]

    return bary[:, 0].unsqueeze(-1) * v0 + bary[:, 1].unsqueeze(-1) * v1 + bary[:, 2].unsqueeze(-1) * v2

losses = []
for i in trange(1_000):
    points = generator(theta, z, A)
    casdf.precache_query_device(points)
    casdf.precache_device()

    casdf.query_device(points, closest, bary, distance, index)

    sampled_points = sample_surface(target_points, target_faces, 1000)
    optext.barycentric_closest_points(points, faces, sampled_points, samples_bary, samples_index)

    v0 = points[faces[samples_index, 0]]
    v1 = points[faces[samples_index, 1]]
    v2 = points[faces[samples_index, 2]]

    sampled_closest = samples_bary[:, 0].unsqueeze(-1) * v0 + samples_bary[:, 1].unsqueeze(-1) * v1 + samples_bary[:, 2].unsqueeze(-1) * v2

    loss_to = (points - closest).pow(2).sum(dim=-1).mean()
    loss_from = (sampled_points - sampled_closest).pow(2).sum(dim=-1).mean()
    loss = loss_to + loss_from
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('amplitudes', amps)

# Parameters at the edge of the domain
sys.path.append('../..')
from mlp import *
from configurations import *

def interpolate(features):
    U = torch.linspace(0, 1, n_theta, device='cuda')
    V = torch.linspace(0, 1, n_z, device='cuda')
    U, V = torch.meshgrid(U, V)
    U = U.reshape(-1).unsqueeze(-1)
    V = V.reshape(-1).unsqueeze(-1)

    return (1 - U) * (1 - V) * features[0] + \
        (1 - U) * V * features[1] + \
        U * (1 - V) * features[2] + \
        U * V * features[3]

mlp = MLP_Positional_LeakyReLU_Encoding().cuda()
features = torch.randn((4, POINT_ENCODING_SIZE), device='cuda', requires_grad=True)

# Train the network to equal
# In other words, initialization
optimizer = torch.optim.Adam(list(mlp.parameters()) + [ features ], lr=1e-2)
for i in trange(1_000):
    points = generator(theta, z, A)
    lerped_features = interpolate(features)
    vertices = mlp(points=points, features=lerped_features)

    loss = (vertices - points).pow(2).sum(dim=-1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Now optimize towards the target geometry
optimizer = torch.optim.Adam(list(mlp.parameters()) + [ features ], lr=1e-2)
for i in trange(1_000):
    points = generator(theta, z, A)
    lerped_features = interpolate(features)
    vertices = mlp(points=points, features=lerped_features)

    casdf.precache_query_device(points)
    casdf.precache_device()

    casdf.query_device(points, closest, bary, distance, index)
    
    sampled_points = sample_surface(target_points, target_faces, 1000)
    optext.barycentric_closest_points(vertices, faces, sampled_points, samples_bary, samples_index)

    v0 = vertices[faces[samples_index, 0]]
    v1 = vertices[faces[samples_index, 1]]
    v2 = vertices[faces[samples_index, 2]]

    sampled_closest = samples_bary[:, 0].unsqueeze(-1) * v0 + samples_bary[:, 1].unsqueeze(-1) * v1 + samples_bary[:, 2].unsqueeze(-1) * v2

    loss_to = (vertices - closest).pow(2).sum(dim=-1).mean()
    loss_from = (sampled_points - sampled_closest).pow(2).sum(dim=-1).mean()
    loss = loss_to + loss_from
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

import seaborn as sns

sns.set()
plt.plot(losses)
plt.yscale('log')
plt.show()

ps.init()
ps.register_surface_mesh('faces', points.detach().cpu().numpy(), faces.cpu().numpy())
ps.register_surface_mesh('target', target_points.cpu().numpy(), target_faces.cpu().numpy())
ps.register_surface_mesh('generated', vertices.detach().cpu().numpy(), faces.cpu().numpy())
ps.show()

