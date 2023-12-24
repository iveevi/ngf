# Packages for training
import imageio.v2 as imageio
import sys
import torch
import optext
import polyscope as ps
import numpy as np

sys.path.append('../../source')
from mesh import load_mesh

def reindex(vertices, faces):
    new_vertices = []
    new_faces = []

    used = {}
    for f in faces:
        new_f = []
        for i in f:
            if i not in used:
                used[i] = len(new_vertices)
                new_vertices.append(vertices[i])

            new_f.append(used[i])

        new_faces.append(new_f)

    new_vertices = np.array(new_vertices)
    new_faces = np.array(new_faces)

    return torch.from_numpy(new_vertices), torch.from_numpy(new_faces)

def parametrize(vertices, faces):
    import matplotlib as plt
    import matplotlib.tri as tri

    uvs = torch.rand((vertices.shape[0], 2))
    print('uvs =', uvs)

    u, v = uvs[:, 0], uvs[:, 1]
    print('u,v', u.shape, v.shape)

    triangulation = tri.Triangulation(u, v, faces)

if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Usage: python main.py <mesh> <count>'

    path = sys.argv[1]
    count = int(sys.argv[2])

    mesh, _ = load_mesh(path)
    mesh.vertices = mesh.vertices.cpu()
    mesh.faces = mesh.faces.cpu()

    face_count = mesh.faces.shape[0]
    seeds = list(torch.randint(0, face_count, (count, )).numpy())
    clusters = optext.cluster_geometry(mesh.optg, seeds, 10)

    patches = []
    for c in clusters:
        V, F = reindex(mesh.vertices, mesh.faces[c])
        patches.append((V, F))

    for V, F in patches:
        parametrize(V, F)

    # ps.init()
    #
    # for i, (V, F) in enumerate(patches):
    #     ps.register_surface_mesh('cluster {}'.format(i), V.numpy(), F.numpy())
    #
    # ps.show()
