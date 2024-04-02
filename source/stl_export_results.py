import os
import glob
import trimesh
import torch
import numpy as np
import optext

from ngf import *
from util import *
from mesh import *

directory = os.path.dirname(__file__)
results = os.path.join(directory, os.path.pardir, 'results')

stl = os.path.join(results, 'stl')
os.makedirs(stl, exist_ok=True)

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

for root, dirs, files in os.walk(results):
    if root != results:
        continue

    for dir in dirs:
        if dir == 'stl':
            continue

        print(os.path.abspath(dir))
        dir_stl = os.path.join(stl, dir)
        os.makedirs(dir_stl, exist_ok=True)

        dir = os.path.join(root, dir)
        for file in glob.glob(dir + '/*.pt'):
            print('\t', os.path.abspath(file))

            ngf = torch.load(file)
            ngf = load_ngf(ngf)

            file = os.path.basename(file)
            file = file.split('.')[0] + '.stl'
            file = os.path.join(dir_stl, file)
            print('\t', os.path.abspath(file))
            
            uvs = ngf.sample_uniform(16)
            V = ngf.eval(*uvs).detach()

            base = ngf.base(16).detach()
            cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, 16)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], 16)
            indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], 16)
            F = remap.remap_device(indices)

            mesh = trimesh.Trimesh(vertices=V.cpu(), faces=F.cpu())
            print('\t', mesh)

            mesh.export(file)

        # TODO: also export the target surface, qslimmed, nvdiffed and draco-ed
