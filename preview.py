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

# eval_vertices = model(args)[0].cpu().detach().numpy()

def chamfer_distance(ref, X):
    extent = np.linalg.norm(np.max(ref, axis=0) - np.min(ref, axis=0))
    sum = 0
    for x in X:
        sum += np.min(np.linalg.norm(ref - x, axis=1)/extent)/X.shape[0]
    for r in ref:
        sum += np.min(np.linalg.norm(X - r, axis=1)/extent)/ref.shape[0]
    return sum

# chamfers = {}
# for i in [ 2 ]:
#     largs = {
#             'points': corner_points,
#             'encodings': corner_encodings,
#             'complexes': complexes,
#             'resolution': i,
#     }
#
#     lv = model(largs)[0].cpu().detach().numpy()
#     chamfers[i] = chamfer_distance(ref.vertices, lv.reshape(-1, 3))

print('-' * 40)
print('Analytics:')
print('-' * 40)

# TODO: python table
total_size /= 1024 * 1024
ref_size /= 1024 * 1024
reduction = (ref_size - total_size) / total_size * 100

print('Total size      {:.3f} MB'.format(total_size))
print('Original size   {:.3f} MB'.format(ref_size))
print('Reduction       {:.3f}%'.format(reduction))
# print('Chamfer (16):  {:.3f}'.format(chamfer))

# for k, v in chamfers.items():
#     print('Chamfer ({:2d})    {:.3f}'.format(k, v))

ps.init()

enable_patch_coloring = False
enable_edge_coloring = False
enable_displacement_coloring = False

def redraw():
    global eval_vertices, downsample_it, upsample_it, resolution, \
        enable_patch_coloring, enable_edge_coloring, enable_displacement_coloring
		
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
    eval = model(args)
    eval_vertices = eval[0].cpu().detach().numpy()
    eval_displacements = eval[1].cpu().detach().numpy()

    ps.register_surface_mesh("ref", ref.vertices, ref.faces)
    for i, (vertices, displacements) in enumerate(zip(eval_vertices, eval_displacements)):
        N = vertices.shape[0]
        triangles = quad_indices(N)
        g = ps.register_surface_mesh("gim{}".format(i), vertices.reshape(-1, 3), triangles)

        displacements = displacements.reshape(-1, 3)
        displacements = np.linalg.norm(displacements, axis=1)

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
            enable_patch_coloring, enable_edge_coloring, enable_displacement_coloring

    changed = False
    if imgui.Button("Increase Resolution"):
        resolution *= 2
        changed = True

    imgui.SameLine()
    if imgui.Button("Decrease Resolution"):
        resolution //= 2
        resolution = max(1, resolution)
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
