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

model = data_dir + '/model.bin'
model = torch.load(model)

complexes = data_dir + '/complexes.bin'
complexes = torch.load(complexes)

corner_points = data_dir + '/points.bin'
corner_points = torch.load(corner_points)

corner_encodings = data_dir + '/encodings.bin'
corner_encodings = torch.load(corner_encodings)

ref = data_dir + '/ref.obj'
ref = trimesh.load(ref)
print('Reference model loaded:', ref.vertices.shape, ref.faces.shape)

print('complexes:', complexes.shape)
print('corner_points:', corner_points.shape)
print('corner_encodings:', corner_encodings.shape)

ps.init()

resolution = 16
args = {
        'points': corner_points,
        'encodings': corner_encodings,
        'complexes': complexes,
        'resolution': resolution,
}

eval_vertices = model(args)[0].cpu().detach().numpy()

def redraw():
    global eval_vertices, downsample_it, upsample_it

    print('eval_vertices:', eval_vertices.shape)
    for i, vertices in enumerate(eval_vertices):
        N = vertices.shape[0]
        triangles = indices(N)
        ps.register_surface_mesh("gim{}".format(i), vertices.reshape(-1, 3), triangles)

def callback():
    # Buttons to increase and decrease the number of iterations
    global eval_vertices, downsample_it, upsample_it, resolution

    changed = False
    if imgui.Button("Increase Resolution"):
        resolution *= 2
        changed = True
    
    imgui.SameLine()
    if imgui.Button("Decrease Resolution"):
        resolution //= 2
        resolution = max(1, resolution)
        changed = True

    if changed:
        args['resolution'] = resolution
        eval_vertices = model(args)[0].cpu().detach().numpy()

        redraw()

redraw()

ps.register_surface_mesh("ref", ref.vertices, ref.faces)
ps.set_user_callback(callback)

ps.show()
