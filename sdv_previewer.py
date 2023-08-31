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

if len(sys.argv) < 2:
    print('Usage: python3 sdv_complexes.py <sdv_complexes_file>')
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

for _ in range(complexes_count):
    size = int.from_bytes(sdv_complexes_file.read(4), byteorder='little')

    vertex_count = int.from_bytes(sdv_complexes_file.read(4), byteorder='little')
    assert vertex_count == size * size

    vertices = sdv_complexes_file.read(12 * vertex_count)
    vflat = np.frombuffer(vertices, dtype=np.float32)
    vertices = vflat.reshape((size, size, 3))
    
    tris = indices(size)
    vflat = vflat.reshape((size * size, 3))
    v0 = vflat[tris[:, 0]]
    v1 = vflat[tris[:, 1]]
    v2 = vflat[tris[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v0
    n = np.cross(e0, e1)

    targets.append(vertices)
    target_normals.append(n)

targets = np.array(targets)
target_normals = np.array(target_normals)

resolution = targets.shape[1]

ps.init()

for i, (tg, tn) in enumerate(zip(targets, target_normals)):
    tg = tg.reshape(-1, 3)
    tn = tn.reshape(-1, 3)
    tn_length = np.linalg.norm(tn, axis=1)
    tn /= tn_length[:, None]
    m = ps.register_surface_mesh("target-{}".format(i), tg, indices(resolution))
    # m.add_vector_quantity("normals", tn, defined_on='faces', enabled=True)
    m.add_scalar_quantity("normals-length", tn_length, defined_on='faces', enabled=True)
    m.set_edge_width(1)
    m.set_edge_color([0.0, 0.0, 0.0])

ps.show()
