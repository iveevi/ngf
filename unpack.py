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

# Arguments
assert len(sys.argv) == 3, 'evaluate.py <reference> <directory>'

reference = sys.argv[1]
directory = sys.argv[2]

assert reference is not None
assert directory is not None

# Load all necessary extensions
geom_cpp = load(name="geom_cpp",
        sources=[ "ext/geometry.cpp" ],
        extra_include_paths=[ "glm" ],
        build_directory="build")

print('Loaded all extensions')

# Create a directory specifically for the unpacked meshes
new_directory = os.path.join(directory, 'unpacked')
if os.path.exists(new_directory):
    shutil.rmtree(new_directory)
os.makedirs(new_directory)

# Copy the reference mesh to the new directory
shutil.copyfile(reference, os.path.join(new_directory, 'ref.obj'))

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

        cmap = make_cmap(c, p, lerped_points, rate)
        F, _ = geom_cpp.sdc_weld(c.cpu(), cmap, lerped_points.shape[0], rate)
        F = F.cuda()

        # TODO: use shorted indices...
        vertices = m(points=lerped_points, features=lerped_features)
        print('    > vertices:', vertices.shape)
        print('    > faces:', F.shape)

        m = meshio.Mesh(vertices.detach().cpu().numpy(), [ ('triangle', F.detach().cpu().numpy()) ])

        # Save using the same name as the NSC representation
        meshio.write(os.path.join(new_directory, nsc + '.obj'), m)
