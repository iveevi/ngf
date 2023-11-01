import argparse
import json
import meshio
import os
import sys
import torch
import shutil

from torch.utils.cpp_extension import load
from scripts.geometry import compute_face_normals, compute_vertex_normals

from util import *
from configurations import *

# Arguments
assert len(sys.argv) == 3, 'evaluate.py <reference> <directory>'

reference = sys.argv[1]
directory = sys.argv[2]

assert reference is not None
assert directory is not None

# Load all necessary extensions
if not os.path.exists('build'):
    os.makedirs('build')

optext = load(name='optext',
        sources=[ 'optext.cu' ],
        extra_include_paths=[ 'glm' ],
        build_directory='build',
        extra_cflags=[ '-O3' ],
        extra_cuda_cflags=[ '-O3' ])

print('Loaded optimization extension')

# Create a directory specifically for the unpacked meshes
new_directory = os.path.join(directory, 'unpacked')
if os.path.exists(new_directory):
    shutil.rmtree(new_directory)
os.makedirs(new_directory)

# Copy the reference mesh to the new directory
shutil.copyfile(reference, os.path.join(new_directory, 'ref.obj'))

def shorted_quads(V, C, sample_rate=16):
    quads = []
    for c in range(C.shape[0]):
        offset = c * sample_rate * sample_rate
        for i in range(sample_rate - 1):
            for j in range(sample_rate - 1):
                a = offset + i * sample_rate + j
                c = offset + (i + 1) * sample_rate + j
                b, d = a + 1, c + 1
                quads.append([a, b, d, c])

    return np.array(quads)

# Recursively find all simple meshes and model configurations
for root, dirs, files in os.walk(directory):
    for nsc in dirs:
        if nsc == 'unpacked':
            continue

        print('Loading NSC representation: %s' % nsc)

        data = torch.load(os.path.join(root, nsc, 'model.pt'))

        m = data['model']
        c = data['complexes']
        p = data['points']
        f = data['features']

        print('  > c:', c.shape)
        print('  > p:', p.shape)
        print('  > f:', f.shape)

        lerper = nsc.split('-')[1]
        print('  > lerper: %s' % lerper)

        ker = clerp(lerps[lerper])
        print('  > clerp: %s' % ker)

        # Compute byte size of the representation
        feature_bytes = f.numel() * f.element_size()
        index_bytes = c.numel() * c.element_size()
        model_bytes = sum([ p.numel() * p.element_size() for p in m.parameters() ])
        vertex_bytes = p.numel() * p.element_size()
        total = feature_bytes + index_bytes + model_bytes + vertex_bytes

        rate = 16
        lerped_points, lerped_features = sample(c, p, f, rate, kernel=ker)
        print('    > lerped_points:', lerped_points.shape)
        print('    > lerped_features:', lerped_features.shape)

        # TODO: construct the quad mesh..
        # cmap = make_cmap(c, p, lerped_points, rate)
        # F, _ = optext.sdc_weld(c.cpu(), cmap, lerped_points.shape[0], rate)
        # F = F.cuda()

        # TODO: use shorted indices...
        vertices = m(points=lerped_points, features=lerped_features).detach()

        I = shorted_quads(vertices.cpu().numpy(), c, rate)
        I = torch.from_numpy(I).int()

        cmap = make_cmap(c, p, lerped_points, rate)
        remap = optext.generate_remapper(c.cpu(), cmap, lerped_points.shape[0], rate)
        F = remap.remap(I).cuda()

        m = meshio.Mesh(vertices.detach().cpu().numpy(), [ ('quad', F.cpu().numpy()) ])

        print('    > vertices:', vertices.shape)
        print('    > faces:', F.shape)

        # Save using the same name as the NSC representation
        meshio.write(os.path.join(new_directory, nsc + '.obj'), m)
