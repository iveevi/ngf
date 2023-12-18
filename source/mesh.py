import meshio
import optext
import os
import subprocess
import torch

from dataclasses import dataclass
from typing import Tuple, Callable
from geometry import compute_vertex_normals, compute_face_normals

@dataclass
class Mesh:
    vertices: torch.Tensor
    faces:    torch.Tensor
    normals:  torch.Tensor
    path:     str = ''
    optg:     optext.geometry = None

def load_mesh(path, normalizer=None) -> Tuple[Mesh, Callable[[torch.Tensor], torch.Tensor]]:
    mesh = meshio.read(path)

    v = torch.from_numpy(mesh.points).float().cuda()
    f = torch.from_numpy(mesh.cells_dict['triangle']).int().cuda()

    if normalizer is None:
        min = v.min(0)[0]
        max = v.max(0)[0]
        center = (min + max) / 2
        extent = (max - min).square().sum().sqrt() / 2.0
        normalizer = lambda x: (x - center) / extent

    v = normalizer(v)
    fn = compute_face_normals(v, f)
    vn = compute_vertex_normals(v, f, fn)

    optg = optext.geometry(v.cpu(), f.cpu())

    return Mesh(v, f, vn, os.path.abspath(path), optg), normalizer

def simplify_mesh(mesh, faces, normalizer) -> Mesh:
    BINARY = os.path.join(os.path.dirname(__file__), '..', 'build', 'simplify')

    reduction = faces/mesh.faces.shape[0]
    result = os.path.join(os.path.dirname(mesh.path), 'simplified.obj')

    subprocess.run([BINARY, mesh.path, result, str(reduction)])

    return load_mesh(result, normalizer)[0]
