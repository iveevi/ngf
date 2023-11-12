import json
import meshio
import os
import sys
import torch
import imageio

from torch.utils.cpp_extension import load
from scripts.geometry import compute_face_normals, compute_vertex_normals

from util import *
from configurations import *

# Generate error figures

# Arguments
assert len(sys.argv) >= 2, 'evaluate.py <reference> [results]'

reference = sys.argv[1]
directory = sys.argv[2] if len(sys.argv) >= 3 else 'results'

assert reference is not None
assert os.path.basename(reference) == 'target.obj'
assert directory is None or os.path.exists(directory)

# Load target reference mesh
print('Loading target mesh:', reference)

target_path = os.path.basename(reference)
target = meshio.read(reference)

v_ref = target.points
v_min = np.min(v_ref, axis=0)
v_max = np.max(v_ref, axis=0)
center = (v_min + v_max) / 2.0
extent = np.linalg.norm(v_max - v_min) / 2.0
normalize = lambda x: (x - center) / extent

target.points = normalize(target.points)

# Load all the available models
models = []

if not directory is None:
    for filename in os.listdir(directory):
        # Only parse .pt
        if not filename.endswith('.pt'):
            continue

        print('Loading model:', filename)

        # Load model
        model = torch.load(os.path.join(directory, filename))
        print(model.keys())
        print('kernel:', model['kernel'])

        model['name'] = os.path.splitext(filename)[0]
        model['kernel'] = sampler(kernel=clerp(lerps[model['kernel']]))
        print('sampler:', model['kernel'])

        models.append(model)

import polyscope as ps

ps.init()

resolution = 32
ps.register_surface_mesh('target', target.points, target.cells_dict['triangle'])

def quadify(C, sample_rate=16):
    quads = []
    for c in range(C.shape[0]):
        offset = c * sample_rate * sample_rate
        for i in range(sample_rate - 1):
            for j in range(sample_rate - 1):
                a = offset + i * sample_rate + j
                c = offset + (i + 1) * sample_rate + j
                b, d = a + 1, c + 1
                quads.append([b, d, c, a])

    return np.array(quads)

def instantiate():
    global resolution, models

    for i, m in enumerate(models):
        model = m['model']
        complexes = m['complexes']
        points = m['points']
        features = m['features']
        sampler = m['kernel']

        # Sample points
        LP, LF = sampler(complexes, points, features, resolution)
        V = model(points=LP, features=LF).detach().cpu().numpy()
        print('V = ', V.shape)

        quads = quadify(complexes, resolution)

        V += np.array([ 1.5 * (i + 1), 0.0, 0.0 ])
        ps.register_surface_mesh(m['name'], V, quads)

def callback():
    import polyscope.imgui as imgui

    global resolution
    print('Callback, resolution:', resolution)

    # if imgui.Button('Dump camera data'):
        # print('Camera data:', ps.camera)

    if imgui.Button('Reduce resolution'):
        resolution = max(2, resolution // 2)
        instantiate()

    if imgui.Button('Increase resolution'):
        resolution = min(32, resolution * 2)
        instantiate()

ps.set_user_callback(callback)

instantiate()
ps.show()
