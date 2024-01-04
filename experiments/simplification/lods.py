import pymeshlab
import sys
import os

assert len(sys.argv) == 2, 'Usage: python lods.py <target>'
assert os.path.exists(sys.argv[1]), f'File {sys.argv[1]} does not exist'

file = sys.argv[1]
directory = os.path.dirname(file)

print(f'Generating lod quadrangulations for {file}')

ms = pymeshlab.MeshSet()
ms.load_new_mesh(file)

ms.save_current_mesh('mesh_original.obj')

ms.meshing_decimation_quadric_edge_collapse(
    targetfacenum=5000,
    # qualitythr=1.0,
    # preservenormal=True,
    # preservetopology=True,
    # preserveboundary=True,
    # optimalplacement=True,
)

ms.save_current_mesh('mesh_simplified.obj')

ms.meshing_repair_non_manifold_edges()
ms.meshing_tri_to_quad_by_smart_triangle_pairing()

ms.save_current_mesh('mesh_quads.obj')
