import pymeshlab
import sys
import os

if len(sys.argv) != 3:
    print("Usage: python quadrangulate.py <input> <faces>")
    sys.exit(1)

file = sys.argv[1]
basename = os.path.basename(file)
basename = os.path.splitext(basename)[0]
output = basename + "_quad.obj"
faces = int(sys.argv[2])

print("Processing mesh file   ", file)
print("Decimation face count  ", str(faces))

ms = pymeshlab.MeshSet()
ms.load_new_mesh(file)
ms.meshing_decimation_quadric_edge_collapse(
    targetfacenum=faces,
    qualitythr=0.7,
    preservenormal=True,
    preservetopology=True,
    preserveboundary=True,
    optimalplacement=False,
)

ms.meshing_repair_non_manifold_edges()
# ms.meshing_tri_to_quad_by_smart_triangle_pairing()
ms.meshing_tri_to_quad_by_4_8_subdivision()
ms.save_current_mesh(output)

print("Saved results to file  ", output)
print("Resulting face count   ", str(ms.current_mesh().face_number()))
print("Resulting vertex count ", str(ms.current_mesh().vertex_number()))
