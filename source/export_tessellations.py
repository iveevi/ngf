import os
import sys
import glob
import torch
import ngfutil
import trimesh
import argparse
import numpy as np

from tqdm import tqdm

from ngf import *
from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngf', type=str)
    parser.add_argument('--directory', type=str)

    args = parser.parse_args(sys.argv[1:])

    basename = os.path.basename(args.ngf)
    basename = basename.split('.')[0]
    print('BASENAME', basename)

    ngf = NGF.from_pt(args.ngf)
    for rate in range(2, 16 + 1):
        uvs = ngf.sample_uniform(rate)
        vertices = ngf.eval(*uvs).detach()
        base = ngf.base(rate).detach()
        cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
        remap = ngfutil.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
        faces = ngfutil.triangulate_shorted(vertices, ngf.complexes.shape[0], rate)
        faces = remap.remap_device(faces)

        destination = basename +  f'-r{rate}.stl'
        print(f'EXPORTING RATE {rate} AS {destination}')
        mesh = trimesh.Trimesh(vertices=vertices.cpu(), faces=faces.cpu())
        mesh.export(os.path.join(args.directory, destination))
