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

from models import *

if len(sys.argv) < 3:
    print('Usage: python3 nsc.py <sdv_complexes_file> <output_dir>')
    sys.exit(1)

sdv_complexes_file = sys.argv[1]
sdv_complexes_file = open(sdv_complexes_file, 'rb')

filename_length = int.from_bytes(sdv_complexes_file.read(4), byteorder='little')
print('Filename length:', filename_length)

filename = sdv_complexes_file.read(filename_length).decode('utf-8')
print('Filename:', filename)

point_count = int.from_bytes(sdv_complexes_file.read(4), byteorder='little')
points = sdv_complexes_file.read(12 * point_count)
points = np.frombuffer(points, dtype=np.float32)
points = points.reshape((point_count, 3))
print('Points:', points.shape)

complexes_count = int.from_bytes(sdv_complexes_file.read(4), byteorder='little')
complexes = sdv_complexes_file.read(16 * complexes_count)
complexes = np.frombuffer(complexes, dtype=np.int32).astype(np.int64)
complexes = complexes.reshape((complexes_count, 4))
print('Complexes:', complexes.shape)

targets = []
target_normals = []
average_edge_length = 0.0

for _ in range(complexes_count):
    size = int.from_bytes(sdv_complexes_file.read(4), byteorder='little')

    vertex_count = int.from_bytes(sdv_complexes_file.read(4), byteorder='little')
    assert vertex_count == size * size

    vertices = sdv_complexes_file.read(12 * vertex_count)
    vflat = np.frombuffer(vertices, dtype=np.float32)
    vertices = vflat.reshape((size, size, 3))

    # Compute the expected normals in each triangle
    tris = indices(size)

    vflat = vflat.reshape((size * size, 3))
    v0 = vflat[tris[:, 0]]
    v1 = vflat[tris[:, 1]]
    v2 = vflat[tris[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v0
    n = np.cross(e0, e1)

    # Require normals to be non-zero
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = n/length

    # Replace NaNs with zeros
    n[np.isnan(n)] = 0.0

    # Abort on NaNs
    # if np.sum(np.isnan(n)) > 0:
    #     raise Exception('NaN normal')

    # Accumulate the average edge length
    average_edge_length += np.mean(np.linalg.norm(e0, axis=1))
    average_edge_length += np.mean(np.linalg.norm(e1, axis=1))

    targets.append(vertices)
    target_normals.append(n.reshape((-1, 3)))

targets = np.array(targets)
target_normals = np.array(target_normals)
average_edge_length /= (2 * complexes_count)

# ps.init()
#
# for i, (vs, ns) in enumerate(zip(targets, target_normals)):
#     vs = vs.reshape(-1, 3)
#     tris = indices(size)
#     ps_mesh = ps.register_surface_mesh('mesh_{}'.format(i), vs, tris)
#     ps_mesh.add_vector_quantity('normals', ns, defined_on='faces', enabled=True)
#
# ps.show()

print('Targets:', targets.shape)
print('Normals:', target_normals.shape)
print('Average edge length:', average_edge_length)

print(complexes)

corner_points = torch.from_numpy(points).cuda()
corner_encodings = torch.zeros((point_count, ENCODING_SIZE), requires_grad=True, device='cuda')

# TODO: its also possible to encoding the lipschitz constants within the points to describe the local curvature?

print(corner_points.shape)
print(corner_encodings.shape)

complexes = torch.from_numpy(complexes).cuda()
targets = torch.from_numpy(targets).cuda()
target_normals = torch.from_numpy(target_normals).cuda()

resolution = targets.shape[1]
args = {
    'points': corner_points,
    'encodings': corner_encodings,
    'complexes': complexes,
    'resolution': resolution
}

nsc = NSC().cuda()

# vertices, d, l = nsc(args)
# vertices -= d
# print(vertices.shape)
#
# ps.init()
#
# for i, vs in enumerate(vertices):
#     vs = vs.reshape(-1, 3)
#     vs = vs.cpu().detach().numpy()
#     ps.register_surface_mesh("complex-{}".format(i), vs, indices(resolution))
#
# ps.show()

tris = torch.from_numpy(indices(resolution)).cuda()
optimizer = torch.optim.Adam(list(nsc.parameters()) + [ corner_encodings ], lr=1e-3)
iterations = 100_000

history = { 'loss': [], 'vertex': [], 'normal': [], 'lipschitz': [] }
for i in tqdm.trange(iterations):
    vertices, d, lipschitz = nsc(args)

    # Target vertex loss
    vertex_loss = torch.mean(torch.pow(vertices - targets, 2))

    # Normal loss
    vs = vertices.reshape(vertices.shape[0], -1, 3)
    v0 = vs[:, tris[:, 0]]
    v1 = vs[:, tris[:, 1]]
    v2 = vs[:, tris[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v0
    normals = torch.cross(e0, e1)
    normals = torch.nn.functional.normalize(normals, dim=2)
    normal_loss = torch.mean(torch.pow(normals - target_normals, 2))

    # TODO: lipschitz?
    loss = vertex_loss # + average_edge_length * normal_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    history['loss'].append(loss.item())
    history['vertex'].append(vertex_loss.item())
    history['normal'].append(normal_loss.item())
    history['lipschitz'].append(lipschitz.item())

    # if i == iterations // 3:
    #     # Only train the network
    #     optimizer = torch.optim.Adam(nsc.parameters(), lr=1e-3)
    #
    # if i == 2 * iterations // 3:
    #     # Only train the encodings
    #     optimizer = torch.optim.Adam([ corner_encodings ], lr=1e-3)

# Save plot
sns.set()
plt.figure(figsize=(20, 10))
plt.plot(history['loss'], label='loss')
plt.plot(history['vertex'], label='vertex')
plt.plot(history['normal'], label='normal')
plt.plot(history['lipschitz'], label='lipschitz')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss')
plt.yscale('log')

# Display results
vertices, d, l = nsc(args)
print(vertices.shape)

ps.init()

for i, vs in enumerate(vertices):
    vs = vs.reshape(-1, 3)
    vs = vs.cpu().detach().numpy()
    ps.register_surface_mesh("complex-{}".format(i), vs, indices(resolution))

ps.show()

# Save the state
output_dir = sys.argv[2]
model_file = os.path.join(output_dir, 'model.bin')
complexes_file = os.path.join(output_dir, 'complexes.bin')
corner_points_file = os.path.join(output_dir, 'points.bin')
corner_encodings_file = os.path.join(output_dir, 'encodings.bin')

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir)
torch.save(nsc, model_file)
torch.save(complexes, complexes_file)
torch.save(corner_points, corner_points_file)
torch.save(corner_encodings, corner_encodings_file)
plt.savefig(os.path.join(output_dir, 'loss.png'))

# TODO: preserve the extension
extension = os.path.splitext(filename)[1]
print('Copying', filename, 'to', os.path.join(output_dir, 'ref' + extension))
shutil.copyfile(filename, os.path.join(output_dir, 'ref' + extension))
