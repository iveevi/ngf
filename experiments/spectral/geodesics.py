import potpourri3d as pp3d
import polyscope as ps
import numpy as np

V, F = pp3d.read_mesh('../../meshes/nefertiti/target.obj')
path_solver = pp3d.EdgeFlipGeodesicSolver(V, F)

# path_pts = path_solver.find_geodesic_path(v_start=14, v_end=200)
# path_pts = path_solver.find_geodesic_path_poly([14, 1000, 200])
path_pts = path_solver.find_geodesic_loop([14, 1000, 200])
print('path_pts', path_pts)

edges = []
for i in range(len(path_pts) - 1):
    edges.append((i, i + 1))
edges = np.array(edges)

ps.init()
ps.register_surface_mesh('surface', V, F)
ps.register_curve_network('geodesic', path_pts, edges)
ps.show()
