import meshio
import os
import subprocess
import torch
import ngfutil

from dataclasses import dataclass
from typing import Tuple, Callable

from .geometry import compute_vertex_normals, compute_face_normals

@dataclass
class Mesh:
    vertices: torch.Tensor
    faces:    torch.Tensor
    normals:  torch.Tensor
    path:     str = ''
    optg:     ngfutil.geometry = None


def mesh_from(V, F) -> Mesh:
    Fn = compute_face_normals(V, F)
    Vn = compute_vertex_normals(V, F, Fn)

    optg = ngfutil.geometry(V.cpu(), F.cpu())

    return Mesh(V, F, Vn, 'raw', optg)


# TODO: load triangle mesh
def load_mesh(path, normalizer=None) -> Tuple[Mesh, Callable[[torch.Tensor], torch.Tensor]]:
    mesh = meshio.read(path)

    v = torch.from_numpy(mesh.points[:, :3]).float().cuda()

    f = None
    if 'triangle' in mesh.cells_dict:
        f = torch.from_numpy(mesh.cells_dict['triangle']).int().cuda()
    else:
        f = torch.from_numpy(mesh.cells_dict['quad']).int().cuda()

    if normalizer is None:
        min, max = v.min(), v.max()
        vmin, vmax = v.min(dim=0)[0], v.max(dim=0)[0]
        scale = (max - min).abs()/2
        center = (vmin + vmax)/2
        normalizer = (lambda x: (x - center)/scale)

    v = normalizer(v)
    fn = compute_face_normals(v, f)
    vn = compute_vertex_normals(v, f, fn)

    if f.shape[1] != 3:
        return Mesh(v, f, vn, os.path.abspath(path), None), normalizer
    else:
        optg = ngfutil.geometry(v.cpu(), f.cpu())
        return Mesh(v, f, vn, os.path.abspath(path), optg), normalizer
