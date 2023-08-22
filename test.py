import numpy as np
import sys
import robust_laplacian
import polyscope as ps
import mesh

from scipy.sparse import linalg as sla

# Get input mesh
if len(sys.argv) < 2:
    print("Usage: python test.py <input_mesh>")
    sys.exit(1)

# Load mesh
m = mesh.load(sys.argv[1])

print(type(m.vertices))
print(type(m.triangles))

print('shape of vertices: ', m.vertices.shape)
print('shape of triangles: ', m.triangles.shape)

print("Input mesh has %d vertices and %d faces" % (m.vertices.shape[0], m.triangles.shape[0]))

m = m.deduplicate()
print("Deduplicated mesh has %d vertices and %d faces" % (m.vertices.shape[0], m.triangles.shape[0]))

qaccel = mesh.cas(m, 128)
print('qaccel: ', qaccel)

# Construct discrete laplacian
L, M = robust_laplacian.mesh_laplacian(m.vertices, m.triangles)

print("Laplacian has %d rows and %d columns" % (L.shape[0], L.shape[1]))

# Compute some eigenvectors
n_eig = 100
evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)
print("Eigenvalues: ", evals)
print("Eigenvectors: ", evecs.shape)

print('shape of evecs: ', evecs[:, 10].shape)

values = evecs[:, 50]
critical = m.critical(values)
print('critical points: ', critical.shape)

max_separation = np.max(m.vertices) - np.min(m.vertices)
print('max separation: ', max_separation)

critical_points = m.vertices[critical, :]
critical_values = values[critical]

print('critical values: ', critical_values.shape)
distances = np.linalg.norm(critical_points[:, None, :] - critical_points[None, :, :], axis=2)

print('dist: ', distances.shape)
threshold_d = 0.1 * max_separation

less_d = np.tril(distances < threshold_d, k=-1)
less = less_d

print('less: ', less.shape)
print(less)

print('Number of close points: ', np.sum(less, axis=0))

# Remove one of the two points
critical = np.delete(critical, np.where(np.sum(less, axis=0) > 0), axis=0)
print('critical points: ', critical.shape)

critical_points = m.vertices[critical, :]

# Convex hull of the points
from scipy.spatial import ConvexHull

hull = ConvexHull(critical_points)
print('hull: ', hull.simplices)

# Visualize
ps.init()

pm = ps.register_surface_mesh("mesh", m.vertices, m.triangles)
for i in range(0, n_eig):
    pm.add_scalar_quantity("eigenvector_%d" % i, evecs[:, i], defined_on='vertices', enabled=False)

ps.register_point_cloud("critical_points", critical_points)
ps.register_surface_mesh("triangles", critical_points, hull.simplices)

ps.show()
