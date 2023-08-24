import numpy as np
import polyscope as ps
import sys
import tqdm
import trimesh

if len(sys.argv) < 2:
    print('Usage: python cut.py <path to .quads file>')
    exit()

with open(sys.argv[1], 'rb') as f:
    length = int.from_bytes(f.read(4), byteorder='little')
    original = f.read(length).decode('utf-8')
    print('Original file:', original)

    vertex_count = int.from_bytes(f.read(4), byteorder='little')
    triangle_count = int.from_bytes(f.read(4), byteorder='little')

    vertices = f.read(vertex_count * 12)
    triangles = f.read(triangle_count * 12)

    vertices = np.frombuffer(vertices, dtype=np.float32).reshape((vertex_count, 3))
    triangles = np.frombuffer(triangles, dtype=np.uint32).reshape((triangle_count, 3))

    print('Vertices:', vertices.shape)
    print('Triangles:', triangles.shape)

    quad_count = int.from_bytes(f.read(4), byteorder='little')

    quads = f.read(quad_count * 8)
    quads = np.frombuffer(quads, dtype=np.uint32).reshape((quad_count, 2))
    print('Quads:', quads.shape)

original = trimesh.load_mesh(original)
opt = trimesh.Trimesh(vertices=vertices, faces=triangles)

opt_normals = np.zeros((opt.vertices.shape[0], 3))
for i in range(opt.faces.shape[0]):
    t = opt.faces[i]
    v0 = opt.vertices[t[0]]
    v1 = opt.vertices[t[1]]
    v2 = opt.vertices[t[2]]

    n = np.cross(v1 - v0, v2 - v0)
    n /= np.linalg.norm(n)

    opt_normals[t[0]] += n
    opt_normals[t[1]] += n
    opt_normals[t[2]] += n

# opt_normals /= np.linalg.norm(opt_normals, axis=1).reshape((-1, 1))

for i in range(opt_normals.shape[0]):
    n = opt_normals[i]
    v = opt.vertices[i]

    n /= np.linalg.norm(n)

    dist = np.linalg.norm(original.vertices - v, axis=1)
    closest = np.argmin(dist)

    on = original.vertex_normals[closest]
    if np.dot(n, on) < 0:
        n *= -1

    opt_normals[i] = n

# For each opt vertex find the closest original vertex
# and assign the original vertex's normal to the opt vertex

# for i in range(opt.vertices.shape[0]):
#     v = opt.vertices[i]
#     dist = np.linalg.norm(original.vertices - v, axis=1)
#     closest = np.argmin(dist)
#
#     opt_normals[i] = original.vertex_normals[closest]

# Extrude each quad into a prism along the normal
prism_vertices = np.zeros((quads.shape[0] * 8, 3))
prism_faces = np.zeros((quads.shape[0] * 12, 3), dtype=np.uint32)
prism_meshes = []

def array_order(t0, t1):
    t0, t1 = set(list(t0)), set(list(t1))
    # print(t0, t1)
    shared = set.intersection(t0, t1)
    unique = set.union(t0, t1) - shared
    # print(shared, unique)

    shared = list(shared)
    unique = list(unique)

    return np.array([ unique[0], shared[0], shared[1], unique[1] ])

for i in range(quads.shape[0]):
    t0 = opt.faces[quads[i, 0]]
    t1 = opt.faces[quads[i, 1]]

    vs = array_order(t0, t1)
    ns = opt_normals[vs]
    vs = opt.vertices[vs]
    print(vs, ns)

    # Compute normal use the cross of diagonals
    d0 = vs[0] - vs[3]
    d1 = vs[1] - vs[2]

    base = vs

    extruded = base + ns * 1
    base -= ns * 0.1

    vertices = np.zeros((8, 3))
    vertices[0] = base[0]
    vertices[1] = base[1]
    vertices[2] = base[2]
    vertices[3] = base[3]

    vertices[4] = extruded[0]
    vertices[5] = extruded[1]
    vertices[6] = extruded[2]
    vertices[7] = extruded[3]

    # Two triangles per each of the six faces
    faces = np.zeros((12, 3), dtype=np.uint32)

    faces[0] = [0, 1, 2]
    faces[1] = [1, 2, 3]

    faces[2] = [0, 1, 4]
    faces[3] = [1, 4, 5]

    faces[4] = [1, 3, 5]
    faces[5] = [3, 5, 7]

    faces[6] = [2, 3, 6]
    faces[7] = [3, 6, 7]

    faces[8] = [0, 2, 4]
    faces[9] = [2, 4, 6]

    faces[10] = [4, 5, 6]
    faces[11] = [5, 6, 7]

    prism_meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces))

    # Add to the global mesh
    voff = i * 8
    toff = i * 12

    for j in range(8):
        prism_vertices[voff + j] = vertices[j]

    for j in range(12):
        prism_faces[toff + j] = faces[j] + voff

prisms = trimesh.Trimesh(vertices=prism_vertices, faces=prism_faces)

# Compute boolean intersections of each prism with the original mesh
intersections = []

with tqdm.tqdm(total=len(prism_meshes)) as pbar:
    for pm in prism_meshes:
        intersection = trimesh.boolean.intersection([pm, original])
        intersections.append(intersection)
        pbar.update(1)
# intersections.append(trimesh.boolean.intersection([prism_meshes[0], original]))

ps.init()

org_m = ps.register_surface_mesh('original', original.vertices, original.faces)
opt_m = ps.register_surface_mesh('opt', opt.vertices, opt.faces)
# prism_m = ps.register_surface_mesh('prisms', prisms.vertices, prisms.faces)

org_m.add_vector_quantity('vertex_normals', original.vertex_normals, defined_on='vertices')
opt_m.add_vector_quantity('vertex_normals', opt_normals, defined_on='vertices')

opt_face_colors = np.zeros((opt.faces.shape[0], 3))
opt_face_colors[quads[:, 0]] = [1, 0.5, 0.5]
opt_face_colors[quads[:, 1]] = [0.5, 1, 0.5]
opt_m.add_color_quantity('opt_face_colors', opt_face_colors, defined_on='faces')

# for i, pm in enumerate(prism_meshes):
#     ps.register_surface_mesh('prism_{}'.format(i), pm.vertices, pm.faces)

for i, intersection in enumerate(intersections):
    ps.register_surface_mesh('intersection_{}'.format(i), intersection.vertices, intersection.faces)

ps.show()
