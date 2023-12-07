import potpourri3d as pp3d
import polyscope as ps
import numpy as np
import robust_laplacian
import scipy.sparse.linalg as sla
import torch
import optext

V, F = pp3d.read_mesh('../../meshes/planck/target.obj')
L, M = robust_laplacian.mesh_laplacian(V, F)

mesh = optext.geometry(torch.tensor(V, dtype=torch.float32),
                       torch.tensor(F, dtype=torch.int32))

print('Read mesh with', V.shape[0], 'vertices and', F.shape[0], 'faces')
print('Computed Laplacian: L.shape =', L.shape, ', M.shape =', M.shape)

# Compute eigenfunctions
n_eig = 25
evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)
print('Computed', n_eig, 'eigenvectors')

# Find critical points
rings = [ set() for _ in range(V.shape[0]) ]
edges = set()
edges_to_faces = {}
edges_to_edges = {}

for fi, f in enumerate(F):
    for i in range(3):
        rings[f[i]].add(f[(i + 1) % 3])
        rings[f[i]].add(f[(i + 2) % 3])
        e = sorted([ f[i], f[(i + 1) % 3] ])
        edges.add(tuple(e))
        edges_to_faces.setdefault(tuple(e), set()).add(fi)

for e in edges:
    v0, v1 = e

    for v in rings[v0]:
        if v == v1:
            continue

        ne = tuple(sorted([ v, v0 ]))
        edges_to_edges.setdefault(e, set()).add(ne)

    for v in rings[v1]:
        if v == v0:
            continue

        ne = tuple(sorted([ v, v1 ]))
        edges_to_edges.setdefault(e, set()).add(ne)

# Collect critical points
critical_mins = set()
critical_maxs = set()

eig = evecs[:, -1]
for i in range(V.shape[0]):
    is_min = True
    for j in rings[i]:
        if eig[j] < eig[i]:
            is_min = False
            break

    if is_min:
        critical_mins.add(i)
        continue

    is_max = True
    for j in rings[i]:
        if eig[j] > eig[i]:
            is_max = False
            break

    if is_max:
        critical_maxs.add(i)
        continue

print('Found', len(critical_mins), 'minima and', len(critical_maxs), 'maxima')

critical_points = list(critical_mins) + list(critical_maxs)
critical_points = np.array(critical_points)
print('Critical points:', critical_points.shape)

# Prune critical points that are too close
max_vertex = V.max(axis=0)
min_vertex = V.min(axis=0)
print('Extent:', (max_vertex - min_vertex))
extent = (max_vertex - min_vertex).max()
threshold = 0.05 * extent

while True:
    pruned = False
    for i in range(critical_points.shape[0]):
        for j in range(i + 1, critical_points.shape[0]):
            dV = V[critical_points[i]] - V[critical_points[j]]
            if np.linalg.norm(dV) < threshold:
                critical_points[j] = -1
                pruned = True

    critical_points = critical_points[critical_points >= 0]
    if not pruned:
        break

print('Pruned critical points:', critical_points.shape)

# Diffusion
solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
sources = critical_points
# distances = solver.compute_distance_multisource(sources)

distances = []
for s in sources:
    distances.append(solver.compute_distance(s))
    print('Computed distance for source', s)

print('Computed distances')
distance_matrix = np.array(distances)
print('Distance matrix:', distance_matrix.shape)

# Cluster each vertex by largest distance
cluster_indices = np.argmin(distance_matrix, axis=0)
print('Cluster indices:', cluster_indices.shape)

clusters = []
for i in range(cluster_indices.max() + 1):
    clusters.append(np.where(cluster_indices == i)[0])

# Collect boundary edges, from triangles with two distinct cluster indices
boundary_edges = {}
for i in range(F.shape[0]):
    v0, v1, v2 = F[i]
    i0, i1, i2 = cluster_indices[v0], cluster_indices[v1], cluster_indices[v2]
    sis = sorted([ (i0, v0), (i1, v1), (i2, v2) ], key=lambda x: x[0])

    # All vertices in same cluster or all distinct
    if sis[0][0] == sis[2][0]:
        # Same; add boundary edge if it is the boundary of the mesh as a whole
        continue
    if sis[0][0] != sis[1][0] and sis[1][0] != sis[2][0]:
        continue

    if sis[0][0] != sis[1][0]:
        assert sis[1][0] == sis[2][0]
        edge = tuple(sorted([ sis[1][1], sis[2][1] ]))
        boundary_edges.setdefault(sis[1][0], set()).add(edge)
    else:
        assert sis[1][0] != sis[2][0]
        assert sis[0][0] == sis[1][0]
        edge = tuple(sorted([ sis[0][1], sis[1][1] ]))
        boundary_edges.setdefault(sis[0][0], set()).add(edge)

# Centroids of triangles will all distinct cluster indices
cluster_corners = {}

corners = []
for i in range(F.shape[0]):
    v0, v1, v2 = F[i]
    i0, i1, i2 = cluster_indices[v0], cluster_indices[v1], cluster_indices[v2]

    if i0 != i1 and i0 != i2 and i1 != i2:
        # Make sure each vertex is connected to its cluster's boundary
        # TODO: this might be unnecessary and dangerous
        # v0_in, v1_in, v2_in = False, False, False
        # for e in boundary_edges[i0]:
        #     if v0 in e:
        #         v0_in = True
        #         break
        #
        # for e in boundary_edges[i1]:
        #     if v1 in e:
        #         v1_in = True
        #         break
        #
        # for e in boundary_edges[i2]:
        #     if v2 in e:
        #         v2_in = True
        #         break
        #
        # if not v0_in or not v1_in or not v2_in:
        #     continue

        vertices = V[[v0, v1, v2]]
        e0 = vertices[1] - vertices[0]
        e1 = vertices[2] - vertices[0]
        e0 /= np.linalg.norm(e0)
        e1 /= np.linalg.norm(e1)

        centroid = vertices.mean(axis=0)
        normal = np.cross(e0, e1)

        index = len(corners)
        corners.append((centroid, normal))

        # cluster_corners.setdefault(i0, []).append((index, i))
        # cluster_corners.setdefault(i1, []).append((index, i))
        # cluster_corners.setdefault(i2, []).append((index, i))

        cluster_corners.setdefault(i0, set()).add(index)
        cluster_corners.setdefault(i1, set()).add(index)
        cluster_corners.setdefault(i2, set()).add(index)

print('Found', len(corners), 'corners')
# print('Cluster corners:', cluster_corners)

# Combine nearby corners
average_edge_length = 0
for e in edges:
    average_edge_length += np.linalg.norm(V[e[0]] - V[e[1]])
average_edge_length /= len(edges)

print('Average edge length:', average_edge_length)
threshold = 10 * average_edge_length
print('Threshold:', threshold)

merge_sets = []
for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        if np.linalg.norm(corners[i][0] - corners[j][0]) < threshold:
            print('Merging corners', i, j)

            found = False
            for s in merge_sets:
                if i in s or j in s:
                    s.add(i)
                    s.add(j)
                    found = True
                    break

            if not found:
                merge_sets.append(set([ i, j ]))

print('Merge sets:', merge_sets)

merging_vertices = []
for s in merge_sets:
    for i in s:
        merging_vertices.append(corners[i])

merging_vertices = np.array(merging_vertices)

# Perform merges
# erase_indices = []
for s in merge_sets:
    # Pick a representative vertex
    representative = list(s)[0]

    for i in s:
        if i == representative:
            continue

        # erase_indices.append(i)

        # Erase from cluster corners
        for c in cluster_corners:
            if i in cluster_corners[c]:
                cluster_corners[c].remove(i)
                cluster_corners[c].add(representative)

# Erase vertices
# erase_indices = sorted(erase_indices, reverse=True)
# for i in erase_indices:
#     corners.pop(i)

# Sort boundary edges topologically (using edge-to-edge map)
# for c in boundary_edges:
#     sorted_edges = []
#
#     bdy = list(boundary_edges[c])
#
#     # Make sure each vertex occurs at least once; weak boundary cycles
#     vertex_counts = {}
#     for e in bdy:
#         vertex_counts.setdefault(e[0], 0)
#         vertex_counts.setdefault(e[1], 0)
#         vertex_counts[e[0]] += 1
#         vertex_counts[e[1]] += 1
#
#     # Remove vertices that appear only once
#     # for v in list(vertex_counts.keys()):
#     #     if vertex_counts[v] == 1:
#     #         del vertex_counts[v]
#     #
#     #         # Find the edge that contains this vertex
#     #         for e in bdy:
#     #             if not v in e:
#     #                 continue
#     #
#     #             bdy.remove(e)
#     #             if v == e[0]:
#     #                 vertex_counts[e[1]] -= 1
#     #             else:
#     #                 vertex_counts[e[0]] -= 1
#     #
#     # # Ensure that each vertex occurs at least twice
#     # for v in vertex_counts:
#     #     assert vertex_counts[v] >= 2, 'Vertex %d occurs %d times' % (v, vertex_counts[v])
#
#     # Start with some arbitrary edge
#     sorted_bdy = []
#
#     remaining = set(bdy)
#     next = [ bdy[0] ]
#     last = set() # set([ bdy[0][0] ])
#
#     remaining.remove(bdy[0])
#     last.add(bdy[0][0])
#
#     # BFS order
#     while len(next) > 0:
#         ne = next.pop(0)
#         sorted_bdy.append(ne)
#
#         new_last = set()
#         new_edges = set()
#
#         for e in remaining:
#             if e[0] in last:
#                 new_edges.add(e)
#                 new_last.add(e[1])
#             elif e[1] in last:
#                 new_edges.add(e)
#                 new_last.add(e[0])
#
#         for e in new_edges:
#             next.append(e)
#             remaining.remove(e)
#
#         last = new_last
#
#     # print('Sorted boundary:', sorted_bdy)
#     boundary_edges[c] = sorted_bdy

# Build boundary polygons
boundary_polygons = {}
for c in boundary_edges:
    bdy = boundary_edges[c]

    boundary_vertices = []
    boundary_indices = []

    for e in boundary_edges[c]:
        s = len(boundary_vertices)
        boundary_vertices.append(V[e[0]])
        boundary_vertices.append(V[e[1]])
        boundary_indices.append([ s, s + 1 ])

    boundary_vertices = np.array(boundary_vertices)
    boundary_indices = np.array(boundary_indices)

    boundary_polygons[c] = (boundary_vertices, boundary_indices)

# Build cluster polygons
cluster_polygons = {}

import itertools

for c in cluster_corners:
    print('Assembling coarse boundary of cluster', c, cluster_corners[c])

    corner_indices = cluster_corners[c]
    if len(corner_indices) < 3:
        # TODO: in this case, supplement until at least 3 corners
        continue

    # Sort corners using boundary of the cluster
    # bdy = boundary_edges[c]

    # Find the vertex of the corner that is in the cluster
    # corner_indices = [ d[0] for d in data ]

    # Generate all permutations of edges; ie all possible polygons
    corner_perms = list(itertools.permutations(corner_indices))

    # Compute angle sum for each permutation
    def angle_between(v0, v1, v2):
        v0 = v0 - v1
        v2 = v2 - v1
        return np.arccos(np.dot(v0, v2) / (np.linalg.norm(v0) * np.linalg.norm(v2)))

    angle_sums = []
    for p in corner_perms:
        angle_sum = 0
        for i in range(len(p)):
            ni = (i + 1) % len(p)
            n2i = (i + 2) % len(p)

            normal = corners[p[ni]][1]
            v0, v1, v2 = corners[p[i]][0], corners[p[ni]][0], corners[p[n2i]][0]
            angle = angle_between(v0, v1, v2)

            # Project vertices onto the tangent plane
            pv0 = v0 - np.dot(v0, normal) * normal
            pv1 = v1 - np.dot(v1, normal) * normal
            pv2 = v2 - np.dot(v2, normal) * normal
            pangle = angle_between(pv0, pv1, pv2)

            angle_sum += pangle

        angle_sums.append(angle_sum)

    print('  > Angle sums:', len(angle_sums))

    # Choose the permutation with the largest angle sum
    best_perm = np.argmax(angle_sums)
    print('  > Best permutation:', best_perm)
    corner_indices = list(corner_perms[best_perm])
    print('  > Corner indices:', corner_indices)

    # corner_indices = []
    # for i in range(len(data)):
    #     f = data[i][1]
    #     v0, v1, v2 = F[f]
    #     i0, i1, i2 = cluster_indices[v0], cluster_indices[v1], cluster_indices[v2]
    #     if i0 == c:
    #         corner_indices.append((data[i][0], v0))
    #     elif i1 == c:
    #         corner_indices.append((data[i][0], v1))
    #     elif i2 == c:
    #         corner_indices.append((data[i][0], v2))
    #     else:
    #         assert False, 'Corner %d not in cluster %d' % (data[i][0], c)
    #
    # # print('Corner indices:', corner_indices)
    #
    # def bdy_index(v):
    #     # Find the first boundary edge that contains this vertex
    #     for i in range(len(bdy)):
    #         if v in bdy[i]:
    #             return i
    #
    #     assert False, 'Vertex %d not in boundary' % v
    #
    # corner_indices = sorted(corner_indices, key=lambda x: bdy_index(x[1]))
    # print('  > Sorted corners:', corner_indices)

    # indices = np.array([ x[0] for x in corner_indices ])
    indices = corner_indices
    vertices = np.array(corners)[indices][:, 0]
    print('  > Vertices:', vertices.shape)
    indices = np.arange(len(indices))
    indices = np.stack([ indices, (indices + 1) % len(indices) ], axis=1)
    cluster_polygons[c] = (vertices, indices)

ps.init()

enable_point_clusters = False

def draw():
    ps.remove_all_structures()

    ps_mesh = ps.register_surface_mesh("mesh", V, F)
    ps.register_point_cloud("sources", V[sources, :])
    # ps.register_point_cloud("merging_vertices", merging_vertices)
    # ps.register_point_cloud("shitters", np.array(shitters)).set_radius(0.0025)

    if enable_point_clusters:
        for i, c in enumerate(clusters):
            ps.register_point_cloud("cluster_" + str(i), V[c, :]).set_radius(0.0025)

    if len(corners) > 0:
        ps.register_point_cloud("corners", np.array(corners)[:, 0]).set_radius(0.0025)

    for c in boundary_polygons:
        vertices, indices = boundary_polygons[c]
        ps.register_curve_network("boundary_polygon_" + str(c), vertices, indices).set_radius(0.0015)

    for c in cluster_polygons:
        vertices, indices = cluster_polygons[c]
        ps.register_curve_network("cluster_polygon_" + str(c), vertices, indices)

def callback():
    import polyscope.imgui as imgui

    if imgui.Button("Toggle point clusters"):
        global enable_point_clusters
        enable_point_clusters = not enable_point_clusters
        draw()

ps.set_user_callback(callback)
draw()

ps.show()
