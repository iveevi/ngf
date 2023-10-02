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

points = data_dir + '/points.bin'
total_size += os.path.getsize(points)
points = torch.load(points)

encodings = data_dir + '/encodings.bin'
total_size += os.path.getsize(encodings)
encodings = torch.load(encodings)

ref = data_dir + '/ref.obj'
print('Loading reference model:', ref)
ref_size = os.path.getsize(ref)
ref = trimesh.load(ref)
print('Reference model loaded:', ref, ref.vertices.shape, ref.faces.shape)

vmin = np.min(ref.vertices, axis=0)
vmax = np.max(ref.vertices, axis=0)
extent = vmax - vmin
ref.vertices[:, 0] -= 1.5 * extent[0]

print('complexes:', complexes.shape)
print('corner_points:', points.shape)
print('corner_encodings:', encodings.shape)

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

resolution = 4
enable_ref = True
enable_nsc = True
enable_normals = False
enable_boundary = False
enable_patch_coloring = False
enable_edge_coloring = False
enable_displacement_coloring = False

def lerp(X, U, V):
    lp00 = X[:, 0, :].unsqueeze(1) * U.unsqueeze(-1) * V.unsqueeze(-1)
    lp01 = X[:, 1, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * V.unsqueeze(-1)
    lp10 = X[:, 3, :].unsqueeze(1) * U.unsqueeze(-1) * (1.0 - V.unsqueeze(-1))
    lp11 = X[:, 2, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * (1.0 - V.unsqueeze(-1))
    return lp00 + lp01 + lp10 + lp11

def sample(sample_rate):
    U = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    V = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    U, V = torch.meshgrid(U, V)

    corner_points = points[complexes, :]
    corner_normals = normals[complexes, :]
    corner_encodings = encodings[complexes, :]

    U, V = U.reshape(-1), V.reshape(-1)
    U = U.repeat((complexes.shape[0], 1))
    V = V.repeat((complexes.shape[0], 1))

    lerped_points = lerp(corner_points, U, V).reshape(-1, 3)
    lerped_normals = lerp(corner_normals, U, V).reshape(-1, 2)
    lerped_encodings = lerp(corner_encodings, U, V).reshape(-1, POINT_ENCODING_SIZE)

    return lerped_points, lerped_normals, lerped_encodings

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

    LP, LN, LE = sample(resolution)
    print('points:', LP.shape, LP)
    print('normals:', LN.shape, LN)
    print('encodings:', LE.shape, LE)

    eval_vertices = model(LP, LN, LE)
    eval_vertices = eval_vertices.reshape(-1, resolution, resolution, 3)
    assert complexes.shape[0] == eval_vertices.shape[0]
    eval_vertices = eval_vertices.detach().cpu().numpy()

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

    for i, vertices in enumerate(eval_vertices):
        N = vertices.shape[0]
        triangles = quad_indices(N)
        vs = vertices.reshape(-1, 3)
        g = ps.register_surface_mesh("gim{}".format(i), vs, triangles)

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
