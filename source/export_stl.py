import os
import glob
import trimesh
import torch
import numpy as np
import optext

from tqdm import tqdm

from ngf import *
from util import *
from mesh import *

directory = os.path.dirname(__file__)
results = os.path.join(directory, os.path.pardir, 'results')

stl = os.path.join(results, 'stl')
stl_patched = os.path.join(results, 'stl-patched')

os.makedirs(stl, exist_ok=True)
os.makedirs(stl_patched, exist_ok=True)

for root, dirs, files in os.walk(results):
    if root != results:
        continue

    for dir in dirs:
        if dir == 'stl' or dir == 'stl_separable':
            continue

        print('\n' + os.path.abspath(dir), end='\n\n')

        dir_stl = os.path.join(stl, dir)
        dir_stl_patched = os.path.join(stl_patched, dir)

        os.makedirs(dir_stl, exist_ok=True)
        os.makedirs(dir_stl_patched, exist_ok=True)

        dir = os.path.join(root, dir)
        for file in glob.glob(dir + '/*.pt'):
            print('\t', os.path.abspath(file))

            ngf = torch.load(file)
            ngf = load_ngf(ngf)

            uvs = ngf.sample_uniform(16)
            V = ngf.eval(*uvs).detach()

            # Single mesh
            base = ngf.base(16).detach()
            cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, 16)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], 16)
            I = optext.triangulate_shorted(V, ngf.complexes.shape[0], 16)
            F = remap.remap_device(I)

            mesh = trimesh.Trimesh(vertices=V.cpu(), faces=F.cpu())
            print('\t', mesh)

            basename = os.path.basename(file).split('.')[0]
            file = basename + '.stl'
            file_stl = os.path.join(dir_stl, file)
            print('\t', os.path.abspath(file_stl))

            mesh.export(file_stl)

        # TODO: also export the target surface, qslimmed, nvdiffed and draco-ed