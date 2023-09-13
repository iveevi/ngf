import os
import polyscope as ps
import polyscope.imgui as imgui
import sys
import torch
import trimesh

from models import *

# Get source directory as first argument
if len(sys.argv) < 2:
    print('Usage: python view.py <data_dir>')
    sys.exit(1)

data_dir = sys.argv[1]
print('Loading from directory:', data_dir)

total_size = 0
model = data_dir + '/model.bin'
total_size += os.path.getsize(model)
model = torch.load(model)

complexes = data_dir + '/complexes.bin'
total_size += os.path.getsize(complexes)
complexes = torch.load(complexes)

corner_points = data_dir + '/points.bin'
total_size += os.path.getsize(corner_points)
corner_points = torch.load(corner_points)

corner_encodings = data_dir + '/encodings.bin'
total_size += os.path.getsize(corner_encodings)
corner_encodings = torch.load(corner_encodings)

# Find the ref.* file
ref = None
for f in os.listdir(data_dir):
    if f.startswith('ref.'):
        ref = data_dir + '/' + f
        break

print('Loading reference model:', ref)
ref_size = os.path.getsize(ref)
ref = trimesh.load(ref)
print('Reference model loaded:', ref, ref.vertices.shape, ref.faces.shape)

vmin = np.min(ref.vertices, axis=0)
vmax = np.max(ref.vertices, axis=0)
extent = vmax - vmin
ref.vertices[:, 0] -= 1.5 * extent[0]

print('complexes:', complexes.shape)
print('corner_points:', corner_points.shape)
print('corner_encodings:', corner_encodings.shape)

resolution = 16
args = {
        'points': corner_points,
        'encodings': corner_encodings,
        'complexes': complexes,
        'resolution': resolution,
}

# Generate and save the meshes
# TODO: flag
import copy

for res in [ 2, 4, 8, 16 ]:
    a = copy.deepcopy(args)
    a['resolution'] = res
    eval = model(a)
    print('eval:', eval[0].shape, eval[1].shape)

    eval_vertices = eval[0].cpu().detach().numpy()
    eval_vertices = eval_vertices.reshape(-1, 3)

    eval_tris = []

    offset = 0
    for i in range(complexes.shape[0]):
        tris = indices(res)
        tris += offset
        eval_tris.append(tris)
        offset += res * res

    eval_tris = np.concatenate(eval_tris, axis=0)

    m = trimesh.Trimesh(vertices=eval_vertices, faces=eval_tris)
    print('m:', m, m.vertices.shape, m.faces.shape)

    # Save the mesh
    mesh_file = data_dir + '/mesh{}x{}.obj'.format(res, res)
    print('Saving mesh to:', mesh_file)
    m.export(mesh_file)

print('-' * 40)
print('Analytics:')
print('-' * 40)

total_size /= 1024 * 1024
ref_size /= 1024 * 1024
reduction = (ref_size - total_size) / total_size * 100

print('Total size      {:.3f} MB'.format(total_size))
print('Original size   {:.3f} MB'.format(ref_size))
print('Reduction       {:.3f}%'.format(reduction))

ps.init()

enable_ref = True
enable_nsc = True
enable_normals = False
enable_boundary = False
enable_patch_coloring = False
enable_edge_coloring = False
enable_displacement_coloring = False

def redraw():
    global eval_vertices, downsample_it, upsample_it, resolution, \
            enable_ref, \
            enable_nsc, \
            enable_normals, \
            enable_boundary, \
            enable_patch_coloring, \
            enable_edge_coloring, \
            enable_displacement_coloring

    color_wheel = [
        (0.750, 0.250, 0.250),
		(0.750, 0.500, 0.250),
		(0.750, 0.750, 0.250),
		(0.500, 0.750, 0.250),
		(0.250, 0.750, 0.250),
		(0.250, 0.750, 0.500),
		(0.250, 0.750, 0.750),
		(0.250, 0.500, 0.750),
		(0.250, 0.250, 0.750),
		(0.500, 0.250, 0.750),
		(0.750, 0.250, 0.750),
		(0.750, 0.250, 0.500)
    ]

    args['resolution'] = resolution
    args['points'] = corner_points
    args['encodings'] = corner_encodings

    eval = model(args)
    eval_vertices = eval[0].cpu().detach().numpy()
    eval_displacements = eval[1].cpu().detach().numpy()

    if enable_ref:
        r = ps.register_surface_mesh("ref", ref.vertices, ref.faces)
        r.set_color((0.5, 1.0, 0.5))

        if enable_normals:
            v0 = ref.vertices[ref.faces[:, 0], :]
            v1 = ref.vertices[ref.faces[:, 1], :]
            v2 = ref.vertices[ref.faces[:, 2], :]
            normals = np.cross(v1 - v0, v2 - v0)
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            normals = (normals + 1.0) / 2.0
            r.add_color_quantity("normals", normals, defined_on="faces", enabled=True)

    if not enable_nsc:
        return

    # ps.register_point_cloud("points", corner_points.cpu().detach().numpy())

    for i, (vertices, displacements) in enumerate(zip(eval_vertices, eval_displacements)):
        N = vertices.shape[0]
        triangles = quad_indices(N)
        vs = vertices.reshape(-1, 3)
        g = ps.register_surface_mesh("gim{}".format(i), vs, triangles)

        displacements = displacements.reshape(-1, 3)
        displacements = np.linalg.norm(displacements, axis=1)

        if enable_normals:
            v0 = vs[triangles[:, 0], :]
            v1 = vs[triangles[:, 1], :]
            v2 = vs[triangles[:, 2], :]
            normals = np.cross(v1 - v0, v2 - v0)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            normals = (normals + 1.0) / 2.0

            g.add_color_quantity("normals", normals, defined_on="faces", enabled=True)

        if enable_boundary:
            bdy = []
            for j in range(N):
                bdy.append(vertices[j, 0])
            for j in range(N):
                bdy.append(vertices[N - 1, j])
            for j in range(N):
                bdy.append(vertices[N - 1 - j, N - 1])
            for j in range(N):
                bdy.append(vertices[0, N - 1 - j])
            bdy = np.array(bdy)
            bdy = ps.register_curve_network("bdy{}".format(i), bdy, edges='loop')
            bdy.set_color((0.0, 0.0, 0.0))
            bdy.set_radius(0.002)

        if enable_displacement_coloring:
            g.add_scalar_quantity("displacements", displacements, enabled=True, cmap="coolwarm")

        if enable_patch_coloring:
            g.set_color(color_wheel[i % len(color_wheel)])
        else:
            g.set_color((0.5, 0.5, 1.0))

        if enable_edge_coloring:
            g.set_edge_color((0.0, 0.0, 0.0))
            g.set_edge_width(1.0)
        else:
            g.set_edge_color((0.0, 0.0, 0.0))
            g.set_edge_width(0.0)

def callback():
    # Buttons to increase and decrease the number of iterations
    global eval_vertices, downsample_it, upsample_it, resolution, \
            enable_ref, \
            enable_nsc, \
            enable_normals, \
            enable_boundary, \
            enable_patch_coloring, \
            enable_edge_coloring, \
            enable_displacement_coloring

    changed = False
    if imgui.Button("Increase Resolution"):
        resolution *= 2
        changed = True

    imgui.SameLine()
    if imgui.Button("Decrease Resolution"):
        resolution //= 2
        resolution = max(1, resolution)
        changed = True

    if imgui.Button("Toggle Reference"):
        enable_ref = not enable_ref
        print('enable_ref:', enable_ref)
        changed = True

    if imgui.Button("Toggle NSC"):
        enable_nsc = not enable_nsc
        print('enable_nsc:', enable_nsc)
        changed = True

    if imgui.Button("Toggle Normals"):
        enable_normals = not enable_normals
        print('enable_normals:', enable_normals)
        changed = True

    if imgui.Button("Toggle Boundary"):
        enable_boundary = not enable_boundary
        print('enable_boundary:', enable_boundary)
        changed = True

    if imgui.Button("Toggle Patch Coloring"):
        enable_patch_coloring = not enable_patch_coloring
        print('enable_patch_coloring:', enable_patch_coloring)
        changed = True

    if imgui.Button("Toggle Edge Coloring"):
        enable_edge_coloring = not enable_edge_coloring
        print('enable_edge_coloring:', enable_edge_coloring)
        changed = True

    if imgui.Button("Toggle Displacement Coloring"):
        enable_displacement_coloring = not enable_displacement_coloring
        print('enable_displacement_coloring:', enable_displacement_coloring)
        changed = True

    if changed:
        ps.remove_all_structures()
        redraw()

redraw()

ps.set_user_callback(callback)

ps.show()
