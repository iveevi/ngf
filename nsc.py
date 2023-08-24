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
for _ in range(complexes_count):
    size = int.from_bytes(sdv_complexes_file.read(4), byteorder='little')

    vertex_count = int.from_bytes(sdv_complexes_file.read(4), byteorder='little')
    assert vertex_count == size * size

    vertices = sdv_complexes_file.read(12 * vertex_count)
    vertices = np.frombuffer(vertices, dtype=np.float32)
    vertices = vertices.reshape((size, size, 3))

    targets.append(vertices)

targets = np.array(targets)
print('Targets:', targets.shape)

print(complexes)

corner_points = torch.from_numpy(points).cuda()
corner_encodings = torch.randn((point_count, POINT_ENCODING_SIZE), requires_grad=True, device='cuda')

# TODO: its also possible to encoding the lipschitz constants within the points to describe the local curvature?

print(corner_points.shape)
print(corner_encodings.shape)

complexes = torch.from_numpy(complexes).cuda()
targets = torch.from_numpy(targets).cuda()

resolution = targets.shape[1]
args = {
    'points': corner_points,
    'encodings': corner_encodings,
    'complexes': complexes,
    'resolution': resolution
}

srnm = NSC().cuda()
optimizer = torch.optim.Adam(list(srnm.parameters()) + [ corner_encodings ], lr=1e-3)
iterations = 20_000

history = { 'loss': [], 'lipschitz': [] }
for i in tqdm.trange(iterations):
    vertices, lipschitz = srnm(args)
    loss = torch.mean(torch.pow(vertices - targets, 2)) # + 0.1 * torch.pow(lipschitz - 1.0, 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    history['loss'].append(loss.item())
    history['lipschitz'].append(lipschitz.item())

    if i == iterations // 2:
        # Only train the encodings
        optimizer = torch.optim.Adam([ corner_encodings ], lr=1e-3)

# Save plot
sns.set()
plt.figure(figsize=(20, 10))
plt.plot(history['loss'], label='loss')
plt.plot(history['lipschitz'], label='lipschitz')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss')
plt.yscale('log')

# Display results
vertices, _ = srnm(args)
print(vertices.shape)

ps.init()

# for i, tg in enumerate(targets):
#     tg = tg.reshape(-1, 3)
#     tg = tg.cpu().detach().numpy()
#     ps.register_surface_mesh("target-{}".format(i), tg, indices(resolution))

# ps.show()
# ps.remove_all_structures()

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
torch.save(srnm, model_file)
torch.save(complexes, complexes_file)
torch.save(corner_points, corner_points_file)
torch.save(corner_encodings, corner_encodings_file)
plt.savefig(os.path.join(output_dir, 'loss.png'))

# TODO: preserve the extension
shutil.copyfile(filename, os.path.join(output_dir, 'ref.obj'))
