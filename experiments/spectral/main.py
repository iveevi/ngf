import meshio
import polyscope as ps
import robust_laplacian
import scipy.sparse.linalg as sla
import numpy as np

# Read input
# NOTE: first simplify the mesh, and then compute critical points
# mesh = meshio.read('../../meshes/mask/target.obj')
mesh = meshio.read('../../meshes/dragon-statue/simplified.obj')
vertices = mesh.points
faces = mesh.cells_dict['triangle']
print('Read mesh with', vertices.shape[0], 'vertices and', faces.shape[0], 'faces')

# (or for a mesh)
L, M = robust_laplacian.mesh_laplacian(vertices, faces)
print('Computed Laplacian')

# Compute some eigenvectors
n_eig = 200
evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)
print('Computed', n_eig, 'eigenvectors')

# Maximal extent for pruning
vertex_max = vertices.max(axis=0)
vertex_min = vertices.min(axis=0)
extent = (vertex_max - vertex_min).max()
print('Extent:', extent)

# Compile a vertex to vertex adjacency list
# NOTE: this is not the same as the one-ring neighbors
adjacency = [ set() for _ in range(vertices.shape[0]) ]
for f in faces:
    for i in range(3):
        adjacency[f[i]].add(f[(i + 1) % 3])
        adjacency[f[i]].add(f[(i + 2) % 3])

adjacency = [ list(a) for a in adjacency ]

# Get one ring neighbors
def sorted_one_ring(faces):
    # Sort each adjacency to form a one-ring loop
    one_ring = []

    bad = []
    for vi, a in enumerate(adjacency):
        local_adjacency = {}
        for f in faces:
            if vi in f:
                j = 0
                k0 = f[j]
                while k0 == vi:
                    j += 1
                    k0 = f[j]

                k1 = f[j]
                while k1 == vi or k1 == k0:
                    j = (j + 1) % 3
                    k1 = f[j]

                assert k0 != k1
                assert k0 != vi
                assert k1 != vi

                local_adjacency.setdefault(k0, set()).add(k1)
                local_adjacency.setdefault(k1, set()).add(k0)

        manifold = True
        for k, v in local_adjacency.items():
            if len(v) != 2:
                manifold = False
                break

        if not manifold:
            print('  > vertex', vi, 'is not manifold')
            one_ring.append([])
            bad.append(vi)
            continue

        rem = set(a)
        sorted_ring = [ a[0] ]

        print('vi', vi, 'rem', rem, 'sorted ring', sorted_ring)
        print('  > local adjacency', local_adjacency)

        while len(rem) > 0:
            s = sorted_ring[-1]
            rem.remove(s)
            for i in rem:
                if i in local_adjacency[s]:
                    sorted_ring.append(i)
                    break

        print('sorted ring', sorted_ring)

        one_ring.append(sorted_ring)

        # Ensure the ring is closed
        if not sorted_ring[0] in adjacency[sorted_ring[-1]]:
            # Show faces that contain each vertex
            print('  > ring not closed:', sorted_ring[0], 'not in adjacency of', sorted_ring[-1])
            one_ring.append([])
            bad.append(vi)

        # assert sorted_ring[0] in adjacency[sorted_ring[-1]]

    return one_ring, bad

neighbors, bad = sorted_one_ring(faces)
print('Computed one-ring neighbors')
print(neighbors[0])

# neighbors = get_one_ring_neighbors(faces)

# Find critical points
critical_points = []
critical_mins = []
critical_maxs = []
critical_saddles = []

eig = evecs[:, -1]
for i in range(vertices.shape[0]):
    is_min = True
    for j in neighbors[i]:
        if eig[j] < eig[i]:
            is_min = False
            break

    if is_min:
        critical_points.append(i)
        critical_mins.append(i)

    is_max = True
    for j in neighbors[i]:
        if eig[j] > eig[i]:
            is_max = False
            break

    if is_max:
        critical_points.append(i)
        critical_maxs.append(i)

    # p = vertices[i]
    # vertex_signs = []
    # for j in neighbors[i]:
    #     sign = 1 if eig[j] > eig[i] else -1
    #     angle_signs.append((j, sign))

    # Sort by angle
    # angle_signs = sorted(angle_signs, key=lambda x: x[0])

    # Count number of sign changes
    # lower_chains = 0
    # chain_on = False
    # for j in range(len(angle_signs)):
    #     if angle_signs[j][1] == -1:
    #         chain_on = True
    #     elif angle_signs[j][1] == 1 and chain_on:
    #         lower_chains += 1
    #         chain_on = False
    #
    # if lower_chains > 1:
    #     print('Saddle:', lower_chains, angle_signs)
    #     critical_points.append(i)
    #     critical_saddles.append(i)

    # n_sign_changes = 0
    # for j in range(len(angle_signs)):
    #     nj = (j + 1) % len(angle_signs)
    #     if angle_signs[j][1] != angle_signs[nj][1]:
    #         n_sign_changes += 1
    #
    # if n_sign_changes % 2 == 0 and n_sign_changes > 2:
    #     print(n_sign_changes, angle_signs)
    #     critical_points.append(i)
    #     critical_saddles.append(i)

print('Found', len(critical_points), 'critical points')
print('Found', len(critical_mins), 'minima')
print('Found', len(critical_maxs), 'maxima')
print('Found', len(critical_saddles), 'saddles')

critical_points = np.array(critical_points)
critical_mins = np.array(critical_mins)
critical_maxs = np.array(critical_maxs)
critical_saddles = np.array(critical_saddles)

# Clean up noisy critical points by persistence thresholding
# min_eig = eig.min()
# max_eig = eig.max()
# eig_range = max_eig - min_eig
#
# # Clean closeby extrema (of the same type)
# cleaned_critical_points = critical_points.copy()
#
# for i in critical_mins:
#     dists = np.linalg.norm(vertices[i] - vertices[critical_mins], axis=1)
#     indices = np.arange(len(critical_mins))
#     sorted_indices = indices[np.argsort(dists)]
#     sorted_indices = critical_mins[sorted_indices]
#     for j in sorted_indices[1:]:
#         vdiff = np.linalg.norm(vertices[i] - vertices[j])/extent
#         if vdiff > 0.05:
#             continue
#
#         pdiff = np.abs((eig[i] - eig[j]) / eig_range)
#         if pdiff < 0.005:
#             cleaned_critical_points = np.delete(cleaned_critical_points, np.where(cleaned_critical_points == j))
#
# for i in critical_maxs:
#     dists = np.linalg.norm(vertices[i] - vertices[critical_maxs], axis=1)
#     indices = np.arange(len(critical_maxs))
#     sorted_indices = indices[np.argsort(dists)]
#     sorted_indices = critical_maxs[sorted_indices]
#     for j in sorted_indices[1:]:
#         vdiff = np.linalg.norm(vertices[i] - vertices[j])/extent
#         if vdiff > 0.05:
#             continue
#
#         pdiff = np.abs((eig[i] - eig[j]) / eig_range)
#         if pdiff < 0.005:
#             cleaned_critical_points = np.delete(cleaned_critical_points, np.where(cleaned_critical_points == j))
#
# print('Cleaned up to', len(cleaned_critical_points), 'critical points')
#
# cleaned_mins = []
# cleaned_maxs = []
# for i in cleaned_critical_points:
#     if i in critical_mins:
#         cleaned_mins.append(i)
#     if i in critical_maxs:
#         cleaned_maxs.append(i)
#
# cleaned_mins = np.array(cleaned_mins)
# cleaned_maxs = np.array(cleaned_maxs)
#
# # Connect each critical point to its four closest neighbors (of opposite type)
# critical_mins = cleaned_mins
# critical_maxs = cleaned_maxs
#
# edges = []
# for i in critical_mins:
#     dists = np.linalg.norm(vertices[i] - vertices[critical_maxs], axis=1)
#     indices = np.arange(len(critical_maxs))
#     sorted_indices = indices[np.argsort(dists)]
#     sorted_indices = critical_maxs[sorted_indices]
#     for j in sorted_indices[:4]:
#         edges.append((i, j))
#
# for i in critical_maxs:
#     dists = np.linalg.norm(vertices[i] - vertices[critical_mins], axis=1)
#     indices = np.arange(len(critical_mins))
#     sorted_indices = indices[np.argsort(dists)]
#     sorted_indices = critical_mins[sorted_indices]
#     for j in sorted_indices[:4]:
#         edges.append((i, j))
#
# edges = np.array(edges)
# print('Found', edges.shape[0], 'critical edges')
#
# edge_vertices = vertices[edges.flatten(), :]
# edge_indices = np.arange(edges.shape[0] * 2).reshape(edges.shape[0], 2)
# print(edge_vertices.shape, edge_indices.shape, edge_indices)

# for e in edges:
#     pdiff = np.abs((eig[e[0]] - eig[e[1]]) / eig_range)
#     if pdiff < 0.005:
#         cleaned_critical_points = np.delete(cleaned_critical_points, np.where(cleaned_critical_points == e[0]))

# Remove edges that are no longer valid
# cleaned_edges = []
# for e in edges:
#     if e[0] in cleaned_critical_points and e[1] in cleaned_critical_points:
#         cleaned_edges.append(e)
#
# cleaned_edges = np.array(cleaned_edges)
#
# edge_vertices = vertices[cleaned_edges.flatten(), :]
# edge_indices = np.arange(cleaned_edges.shape[0] * 2).reshape(cleaned_edges.shape[0], 2)

# Visualize
ps.init()

ps_mesh = ps.register_surface_mesh('mesh', vertices, faces)
ps_mesh.add_scalar_quantity('eigenvalues', evecs[:, -1], enabled=True)
# ps.register_point_cloud('critical_points', vertices[critical_points, :], enabled=True)

# ps_edge = ps.register_curve_network('critical_edges', edge_vertices, edge_indices, enabled=True)
# ps_edge.set_radius(0.001)

# ps.register_point_cloud('cleaned_critical_points', vertices[cleaned_critical_points, :], enabled=True)
# ps.register_point_cloud('crit_mins', vertices[critical_mins, :], enabled=True)
# ps.register_point_cloud('crit_maxs', vertices[critical_maxs, :], enabled=True)

ps.register_point_cloud('cleaned_mins', vertices[critical_mins, :], enabled=True)
ps.register_point_cloud('cleaned_maxs', vertices[critical_maxs, :], enabled=True)
ps.register_point_cloud('bad', vertices[bad, :], enabled=True)
# ps.register_point_cloud('cleaned_saddles', vertices[critical_saddles, :], enabled=True)

ps.show()
