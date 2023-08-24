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
for _ in range(complexes_count):
    size = int.from_bytes(sdv_complexes_file.read(4), byteorder='little')

    vertex_count = int.from_bytes(sdv_complexes_file.read(4), byteorder='little')
    assert vertex_count == size * size

    vertices = sdv_complexes_file.read(12 * vertex_count)
    vertices = np.frombuffer(vertices, dtype=np.float32)
    vertices = vertices.reshape((size, size, 3))

    targets.append(vertices)

targets = np.array(targets)
resolution = targets.shape[1]

ps.init()

for i, tg in enumerate(targets):
    tg = tg.reshape(-1, 3)
    ps.register_surface_mesh("target-{}".format(i), tg, indices(resolution))

ps.show()
