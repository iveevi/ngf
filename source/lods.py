import pymeshlab
import sys
import os

assert len(sys.argv) == 2, 'Usage: python lods.py <target>'
assert os.path.exists(sys.argv[1]), f'File {sys.argv[1]} does not exist'

file = sys.argv[1]
directory = os.path.dirname(file)
prefix = os.path.join(directory, 'source-lod')

resolutions = [ 5000, 2000, 500, 200 ]

print(f'Generating lod quadrangulations for {file}')
for i, res in enumerate(resolutions):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=res)
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_tri_to_quad_by_smart_triangle_pairing()
    ms.save_current_mesh(prefix + f'{i + 1}.obj')
    print(f'  > generated simplification at {res} faces')
