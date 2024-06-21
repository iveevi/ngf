import os
import json
import tqdm
import torch
import logging
import ngfutil
import argparse
import pymeshlab
import trimesh
import multiprocessing

from util import *
from ngf import NGF
from render import Renderer

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, help='Target mesh')
    parser.add_argument('--lod', type=int, default=2000, help='Number of patches to partition')

    results = os.makedirs('extras/simplified', exist_ok=True)

    args = parser.parse_args()

    m, _ = load_mesh(args.mesh)
    p = 2 * args.lod / m.faces.shape[0]

    logging.info(f'Targetting ratio of {100 * p:.3f}%')

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(args.mesh)

    k = 6
    L = 120
    for i in range(0, L):
        destination = f'extras/simplified/m{i:03d}.stl'

        r = p + (1 - p) * ((L - i)/L) ** k
        if r < p:
            break

        count = int(m.faces.shape[0] * r)

        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=count, qualitythr=1.0)
        logging.info(f'Simplfied to {count} faces ({100 * r:.3f}%)')

        ms.save_current_mesh(destination)
        logging.info(f'Saved result to {destination}')

    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2 * args.lod, qualitythr=1.0)

    ms.save_current_mesh('extras/mfinal.obj')

    ms.meshing_repair_non_manifold_edges()
    ms.meshing_tri_to_quad_by_smart_triangle_pairing()

    ms.save_current_mesh('extras/mquads.obj')
