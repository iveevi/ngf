import math
import os
import polyscope as ps
import polyscope.imgui as imgui
import sys
import torch

from models import *

# Get source directory as first argument
if len(sys.argv) < 2:
    print('Usage: python view.py <data_dir>')
    sys.exit(1)

data_dir = sys.argv[1]
print('Loading from directory:', data_dir)

total_size = 0
model = data_dir + '/model.bin'
total_size += os.path.getsize(model)
model = torch.load(model)

complexes = data_dir + '/complexes.bin'
total_size += os.path.getsize(complexes)
complexes = torch.load(complexes)

corner_points = data_dir + '/points.bin'
total_size += os.path.getsize(corner_points)
corner_points = torch.load(corner_points)

corner_encodings = data_dir + '/encodings.bin'
total_size += os.path.getsize(corner_encodings)
corner_encodings = torch.load(corner_encodings)

print('complexes:', complexes.shape)
print('corner_points:', corner_points.shape)
print('corner_encodings:', corner_encodings.shape)

def delta_omega(n):
    # Find all factors
    min_delta = 1e9
    min_pair = None, None
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            m = n // i
            delta = abs(m - i)
            if delta < min_delta:
                min_delta = delta
                min_pair = i, m

    return min_delta, min_pair

def delta_metric(n):
    min_delta, pair = delta_omega(n)
    delta = abs(n - complexes.shape[0])
    return delta * min_delta, pair

min_metric, min_pair = delta_omega(complexes.shape[0])
for i in range(1, 100):
    metric, pair = delta_omega(complexes.shape[0] + i)
    if metric < min_metric:
        min_metric = metric
        min_pair = pair

resolution = 16
args = {
        'points': corner_points,
        'encodings': corner_encodings,
        'complexes': complexes,
        'resolution': resolution,
}

eval = model(args)
vertices, _, _ = eval
print('vertices:', vertices.shape)

ps.init()

dimx, dimy = min_pair
for c in range(0, complexes.shape[0]):
    vs = vertices[c].detach().cpu().numpy().reshape(-1, 3)
    qs = quad_indices(resolution)

    # UV coorindates
    ix, iy = c % dimx, (c // dimx) % dimy

    u0, v0 = ix / dimx, iy / dimy
    u1, v1 = (ix + 1) / dimx, (iy + 1) / dimy

    U = np.linspace(u0, u1, resolution)
    V = np.linspace(v0, v1, resolution)

    UV = np.stack(np.meshgrid(U, V), axis=-1).reshape(-1, 2)

    # Add dimension for color
    UV = np.concatenate([ UV, np.zeros((UV.shape[0], 1)) ], axis=-1)
    m = ps.register_surface_mesh('mesh' + str(c), vs, qs)
    m.add_color_quantity('uv', UV, defined_on='vertices', enabled=True)

ps.show() 
