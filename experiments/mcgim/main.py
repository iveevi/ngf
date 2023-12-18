# Packages for training
import imageio.v2 as imageio
import sys
import torch
import optext
import polyscope as ps

sys.path.append('../../source')
from mesh import load_mesh

if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Usage: python main.py <mesh> <count>'

    path = sys.argv[1]
    count = int(sys.argv[2])

    # print('Loading mesh from {}...'.format(path))
    mesh, _ = load_mesh(path)
    # print('  > Mesh with {} vertices and {} faces'.format(mesh.vertices.shape[0], mesh.faces.shape[0]))

    face_count = mesh.faces.shape[0]
    seeds = list(torch.randint(0, face_count, (count, )).numpy())

    clusters = optext.cluster_geometry(mesh.optg, seeds, 3)
    # print('  > Cluster sizes: {}'.format([ len(c) for c in clusters ]))

    ps.init()

    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    for i, c in enumerate(clusters):
        ps.register_surface_mesh('cluster {}'.format(i), vertices, faces[c])

    ps.show()
