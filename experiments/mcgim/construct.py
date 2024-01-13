# Packages for training
import imageio.v2 as imageio
import sys
import torch
import optext
import polyscope as ps
import numpy as np
import optext
import nvdiffrast.torch as dr

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

    remap = { v: k for k, v in used.items() }

    return torch.from_numpy(new_vertices), torch.from_numpy(new_faces), remap

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

def expand_ordered_boundary(obdy):
    bdy = []
    for i in range(len(obdy) - 1):
        ni = (i + 1) % len(obdy)
        a, b = obdy[i], obdy[ni]
        bdy.append(tuple(sorted([ a, b ])))
    return bdy

if __name__ == '__main__':
    assert len(sys.argv) == 4, 'Usage: python main.py <mesh> <count> <rate>'

    path = sys.argv[1]
    count = int(sys.argv[2])
    sample_rate = int(sys.argv[3])

    mesh, _ = load_mesh(path)

    mesh.vertices = mesh.vertices.cpu()
    mesh.faces = mesh.faces.cpu()

    print('Mesh', mesh.vertices.shape, mesh.faces.shape)
    V, F = optext.mesh_deduplicate(mesh.vertices, mesh.faces)
    print('Deduplicated', V.shape, F.shape)
    mesh.vertices = V
    mesh.faces = F

    face_count = mesh.faces.shape[0]

    torch.manual_seed(0)
    seeds = list(torch.randint(0, face_count, (count, )).numpy())

    # TODO: use normal flatness metric
    clusters = optext.cluster_geometry(mesh.optg, seeds, 5, 'flat')

    patches = []
    remaps = []
    for c in clusters:
        V, F, remap = reindex(mesh.vertices, mesh.faces[c])
        # print('remap', remap)
        bdy = boundary(F)
        obdy = ordered_boundary(bdy)
        patches.append((V, F, obdy))
        remaps.append(remap)

    uvs = optext.parametrize_multicharts(patches)

    # # Find the shared boundaries between clusters
    # shared = {}
    # for i in range(len(patches)):
    #     for j in range(i + 1, len(patches)):
    #         print('Comparing patches %d and %d' % (i, j))
    #
    #         rma = remaps[i]
    #         rmb = remaps[j]
    #
    #         bdya = expand_ordered_boundary(patches[i][2])
    #         bdyb = expand_ordered_boundary(patches[j][2])
    #
    #         # bdya = [ tuple(sorted([ rma[a], rma[b] ])) for a, b in bdya ]
    #         # bdyb = [ tuple(sorted([ rmb[a], rmb[b] ])) for a, b in bdyb ]
    #
    #         bdya = [ (rma[a], rma[b]) for a, b in bdya ]
    #         bdyb = [ (rmb[a], rmb[b]) for a, b in bdyb ]
    #
    #         # TODO: need to reindex within each patch...
    #
    #         bdya, bdyb = set(bdya), set(bdyb)
    #
    #         # print('  > i bdy', bdya)
    #         # print('  > j bdy', bdyb)
    #
    #         bdy_intersection = bdya.intersection(bdyb)
    #         print('  > intersection', bdy_intersection)

    # for uv, patch in zip(uvs, patches):
    #     vertices, faces = patch[0].numpy(), patch[1].numpy()
    #
    #     import matplotlib.pyplot as plt
    #     import matplotlib.tri as tri
    #
    #     fig, axs = plt.subplots(1, 3)
    #
    #     triangulation = tri.Triangulation(uv[:, 0], uv[:, 1], faces)
    #     for i in range(3):
    #         c = axs[i].tricontourf(triangulation, vertices[:, i], cmap='coolwarm')
    #         axs[i].triplot(triangulation, 'k-', lw=0.25)
    #         axs[i].axis('off')
    #         axs[i].set_aspect('equal')
    #         axs[i].set_title('UVs')
    #
    #         # divider = make_axes_locatable(axs[1, i])
    #         # cax = divider.append_axes('right', size='5%', pad=0.05)
    #         # fig.colorbar(c, cax=cax)
    #
    #     plt.show()


    ctx = dr.RasterizeCudaContext()

    parametrizations = []
    for uv, patch in zip(uvs, patches):
        uv = uv.cuda()

        V = patch[0].cuda()
        F = patch[1].cuda()

        uv = (1 - 0.5/sample_rate) * (2 * uv - 1)
        uv = torch.nn.functional.pad(uv, (0, 2), 'constant', 1.0).unsqueeze(0)

        r = dr.rasterize(ctx, uv, F, (sample_rate, sample_rate))[0]
        v = dr.interpolate(V, r, F)[0][0]

        parametrizations.append((uvs, v))

    # import matplotlib.pyplot as plt
    #
    # N = int(np.sqrt(len(parametrizations)))
    # fig, axs = plt.subplots(N, max(1, (len(parametrizations) + N - 1) // N), layout='constrained')
    # # fig, axs = plt.subplots(1, len(parametrizations), layout='constrained')
    #
    # axs = axs.flatten()
    # for i, (uvs, v) in enumerate(parametrizations):
    #     # v /= torch.linalg.norm(v, axis=-1).unsqueeze(-1)
    #     # print('v = ', v)
    #
    #     axs[i].imshow((0.5 * v + 0.5).cpu().numpy())
    #     axs[i].axis('off')
    #
    # plt.show()

    # ps.init()
    # for k, (uvs, v) in enumerate(parametrizations):
    #     indices = []
    #
    #     v = v.reshape(-1, 3).cpu().numpy()
    #     for i in range(sample_rate - 1):
    #         for j in range(sample_rate - 1):
    #             a = i * sample_rate + j
    #             c = (i + 1) * sample_rate + j
    #             b, d = a + 1, c + 1
    #             indices.append([a, b, c])
    #             indices.append([b, d, c])
    #
    #             vs = v[[a, b, c, d]]
    #             d0 = np.linalg.norm(vs[0] - vs[3])
    #             d1 = np.linalg.norm(vs[1] - vs[2])
    #
    #             if d0 < d1:
    #                 indices.append([a, b, d])
    #                 indices.append([a, d, c])
    #             else:
    #                 indices.append([a, b, c])
    #                 indices.append([b, d, c])
    #
    #     indices = np.array(indices)
    #
    #     ps.register_surface_mesh('patch-%d' % k, v, indices)
    #
    # ps.show()

    # Save the multichart geometry image
    mcgim = []
    for uv, v in parametrizations:
        mcgim.append(v)

    mcgim = torch.stack(mcgim)
    torch.save(mcgim, 'mcgim-%d-%d.pt' % (count, sample_rate))
