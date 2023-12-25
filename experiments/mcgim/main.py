# Packages for training
import imageio.v2 as imageio
import sys
import torch
import optext
import polyscope as ps
import numpy as np
import optext

sys.path.append('../../source')
from mesh import load_mesh

def reindex(vertices, faces):
    new_vertices = []
    new_faces = []

    used = {}
    for f in faces.numpy():
        new_f = []
        for i in f:
            if i not in used:
                used[i] = len(new_vertices)
                new_vertices.append(vertices[i])

            new_f.append(used[i])

        new_faces.append(new_f)

    new_vertices = np.array(new_vertices).astype(np.float32)
    new_faces = np.array(new_faces).astype(np.int32)

    return torch.from_numpy(new_vertices), torch.from_numpy(new_faces)

def adjacency(faces):
    map = {}
    for f in faces.numpy():
        for i in range(3):
            ni = (i + 1) % 3
            a, b = f[i], f[ni]
            map.setdefault(a, set()).add(b)
            map.setdefault(b, set()).add(a)

    return map

def check_boundary(boundary):
    if len(boundary) == 0:
        raise Exception('Null boundary; most likely a degenerate patch')

    remaining = boundary.tolist()
    remaining = set([ (a, b) for a, b in remaining ])

    s = next(iter(remaining))[0]

    queue = [ s ]
    while len(queue) > 0:
        s = queue.pop()

        to_remove = set()
        for e in remaining:
            if e[0] == s:
                queue.append(e[1])
                to_remove.add(e)
            elif e[1] == s:
                queue.append(e[0])
                to_remove.add(e)

        remaining -= to_remove

    return len(remaining) == 0

def ordered_boundary(boundary):
    remaining = boundary.tolist()
    remaining = set([ (a, b) for a, b in remaining ])

    ordered = [ ]

    s = next(iter(remaining))[0]
    while True:
        ordered.append(s)

        p = s
        for e in remaining:
            if e[0] == s:
                s = e[1]
                remaining.remove(e)
                break
            if e[1] == s:
                s = e[0]
                remaining.remove(e)
                break

        if p == s:
            break

    return np.array(ordered).astype(np.int32)

def boundary(faces):
    # Count edges
    edges = {}
    for f in faces.numpy():
        for i in range(3):
            ni = (i + 1) % 3
            a, b = f[i], f[ni]
            e = tuple(sorted([a, b]))
            edges.setdefault(e, 0)
            edges[e] += 1

    # Get boundary edges
    boundary_edges = set()
    for e in edges:
        if edges[e] == 1:
            boundary_edges.add(e)

    b = np.array(list(boundary_edges))
    if not check_boundary(b):
        raise Exception('Bad boundary detected, try increasing number of charts')

    return b

def parametrize(vertices, faces):
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import seaborn as sns

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    sns.set()

    b = boundary(faces)
    o = ordered_boundary(b)

    huvs, uvs = optext.parametrize_chart(vertices, faces, o)

    fig, axs = plt.subplots(2, 3)

    triangulation = tri.Triangulation(huvs[:, 0], huvs[:, 1], faces)
    for i in range(3):
        c = axs[0, i].tricontourf(triangulation, vertices[:, i], cmap='coolwarm')
        axs[0, i].triplot(triangulation, 'k-', lw=0.25)
        axs[0, i].axis('off')
        axs[0, i].set_aspect('equal')
        axs[0, i].set_title('Harmonics')

        # divider = make_axes_locatable(axs[0, i])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(c, cax=cax)

    triangulation = tri.Triangulation(uvs[:, 0], uvs[:, 1], faces)
    for i in range(3):
        c = axs[1, i].tricontourf(triangulation, vertices[:, i], cmap='coolwarm')
        axs[1, i].triplot(triangulation, 'k-', lw=0.25)
        axs[1, i].axis('off')
        axs[1, i].set_aspect('equal')
        axs[1, i].set_title('UVs')

        # divider = make_axes_locatable(axs[1, i])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(c, cax=cax)

    plt.show()

if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Usage: python main.py <mesh> <count>'

    path = sys.argv[1]
    count = int(sys.argv[2])

    mesh, _ = load_mesh(path)

    mesh.vertices = mesh.vertices.cpu()
    mesh.faces = mesh.faces.cpu()

    print('Mesh', mesh.vertices.shape, mesh.faces.shape)
    V, F = optext.mesh_deduplicate(mesh.vertices, mesh.faces)
    print('Deduplicated', V.shape, F.shape)
    mesh.vertices = V
    mesh.faces = F

    face_count = mesh.faces.shape[0]
    seeds = list(torch.randint(0, face_count, (count, )).numpy())

    # TODO: use normal flatness metric
    clusters = optext.cluster_geometry(mesh.optg, seeds, 10, 'flat')

    patches = []
    for c in clusters:
        V, F = reindex(mesh.vertices, mesh.faces[c])
        patches.append((V, F))

    ps.init()
    for i, (V, F) in enumerate(patches):
        ps.register_surface_mesh('cluster {}'.format(i), V.numpy(), F.numpy())
    ps.show()

    for V, F in patches:
        parametrize(V, F)
