import os
import sys
import torch
import optext
import meshio
import shutil
import numpy as np

from util import make_cmap
from ngf import load_ngf
from mesh import load_mesh

def shorted_quads(C, sample_rate=16):
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

# Arguments
if __name__ == '__main__':
    assert len(sys.argv) == 3, 'python unpack.py <target> <results>'

    target_path = sys.argv[1]
    results_path = sys.argv[2]

    assert os.path.exists(target_path), 'Target mesh does not exist'
    assert os.path.exists(results_path), 'Results directory does not exist'

    target, normalizer = load_mesh(target_path)

    unpack_dir = os.path.join(results_path, 'unpacked')
    if os.path.exists(unpack_dir):
        shutil.rmtree(unpack_dir)
    os.makedirs(unpack_dir)

    m = meshio.Mesh(target.vertices.cpu().numpy(), [ ('triangle', target.faces.cpu().numpy()) ])
    meshio.write(os.path.join(unpack_dir, 'target.obj'), m)

    for root, dirs, files in os.walk(results_path):
        for file in files:
            if not file.endswith('.pt'):
                continue

            print('Loading NGF representation: %s' % file)

            data = torch.load(os.path.join(root, file))
            print('data =', data.keys())

            ngf = load_ngf(data)

            lerped_points, lerped_features = ngf.sample(16)
            V = ngf.mlp(points=lerped_points, features=lerped_features).detach()

            I = shorted_quads(ngf.complexes, 16)
            I = torch.from_numpy(I).int()

            cmap = make_cmap(ngf.complexes, ngf.points.detach(), lerped_points.detach(), 16)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, lerped_points.shape[0], 16)
            F = remap.remap(I).cuda()

            m = meshio.Mesh(V.cpu().numpy(), [ ('quad', F.cpu().numpy()) ])
            meshio.write(os.path.join(unpack_dir, file + '.obj'), m)
            
            print('    > vertices:', V.shape)
            print('    > faces:', F.shape)
