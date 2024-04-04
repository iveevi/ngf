import os
import bpy
import numpy as np
import sys
import glob
import io

from contextlib import redirect_stdout

file_directory = os.path.dirname(__file__)
load_directory = os.path.join(file_directory, os.path.pardir)
sys.path += [ file_directory, load_directory ]

from util import *
from ngf import *

def make_cmap(complexes, points, LP, sample_rate):
    Cs = complexes.cpu().numpy()
    lp = LP.detach().cpu().numpy()

    cmap = dict()
    for i in range(Cs.shape[0]):
        for j in Cs[i]:
            if cmap.get(j) is None:
                cmap[j] = set()

        corners = np.array([
            0, sample_rate - 1,
            sample_rate * (sample_rate - 1),
            sample_rate ** 2 - 1
        ]) + (i * sample_rate ** 2)

        qvs = points[Cs[i]].cpu().numpy()
        cvs = lp[corners]

        for j in range(4):
            # Find the closest corner
            dists = np.linalg.norm(qvs[j] - cvs, axis=1)
            closest = np.argmin(dists)
            cmap[Cs[i][j]].add(corners[closest])

    return cmap

import trimesh
import optext
import shutil

path = '/home/venki/projects/ngf/results/lucy/lod2.pt'
directory = '/home/venki/projects/ngf/results/patched'

rotation = np.radians([ 90, 0, -140 ])

ngf = torch.load(path)
ngf = load_ngf(ngf)

shutil.rmtree(directory)
os.makedirs(directory, exist_ok=True)

uvs = ngf.sample_uniform(16)
V = ngf.eval(*uvs).detach()

# Single mesh
base = ngf.base(16).detach()
cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, 16)
remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], 16)
I = optext.triangulate_shorted(V, ngf.complexes.shape[0], 16)
F = remap.remap_device(I)
mesh = trimesh.Trimesh(vertices=V.cpu(), faces=F.cpu())
print('\t', mesh)
mesh.export(os.path.join(directory, 'full.stl'))

# For each resolution
from tqdm import tqdm

colors = [
	(0.880, 0.320, 0.320, 1),
	(0.880, 0.530, 0.320, 1),
	(0.880, 0.740, 0.320, 1),
	(0.810, 0.880, 0.320, 1),
	(0.600, 0.880, 0.320, 1),
	(0.390, 0.880, 0.320, 1),
	(0.320, 0.880, 0.460, 1),
	(0.320, 0.880, 0.670, 1),
	(0.320, 0.880, 0.880, 1),
	(0.320, 0.670, 0.880, 1),
	(0.320, 0.460, 0.880, 1),
	(0.390, 0.320, 0.880, 1),
	(0.600, 0.320, 0.880, 1),
	(0.810, 0.320, 0.880, 1),
	(0.880, 0.320, 0.740, 1),
	(0.880, 0.320, 0.530, 1)
]

def indices(sample_rate):
    triangles = []
    for i in range(sample_rate - 1):
        for j in range(sample_rate - 1):
            a = i * sample_rate + j
            c = (i + 1) * sample_rate + j
            b, d = a + 1, c + 1
            triangles.append([a, b, c])
            triangles.append([b, d, c])
    return np.array(triangles)

for rate in [ 4, 8, 16 ]:
    dirate = os.path.join(directory, f'r{rate:02d}')
    os.makedirs(dirate, exist_ok=True)

    uvs = ngf.sample_uniform(rate)
    V = ngf.eval(*uvs).detach()
    F = indices(rate)

    banks = [ [] for _ in colors ]

    V = V.reshape(-1, rate * rate, 3).cpu()
    for i, Vi in tqdm(enumerate(V)):
        mesh = trimesh.Trimesh(vertices=Vi, faces=F)
        # mesh.export(os.path.join(dirate, f'p{i}.stl'))
        ik = i % len(colors)
        banks[ik].append(mesh)

    for i, bank in enumerate(banks):
        scene = trimesh.Scene(bank)
        scene.export(os.path.join(dirate, f'p{i}.stl'))

# Rendering
C = bpy.context
D = bpy.data
R = bpy.context.scene.render
W = bpy.context.scene.world

D.objects[1].select_set(True)
D.objects[2].select_set(True)
bpy.ops.object.delete()

enode = W.node_tree.nodes.new('ShaderNodeTexEnvironment')
enode.image = bpy.data.images.load('/home/venki/downloads/rural_crossroads_2k.hdr')

node_tree = W.node_tree
node_tree.links.new(enode.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])

from tqdm import tqdm

path = os.path.join(directory, 'full.stl')
basename = os.path.basename(path)
basename = basename.split('.')[0]

bpy.ops.import_mesh.stl(filepath=path)
M = D.objects[basename]

mat = add_material('Main', use_nodes=True, make_node_tree_empty=True)
nodes = mat.node_tree.nodes
links = mat.node_tree.links
output_node = nodes.new(type='ShaderNodeOutputMaterial')
principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')

set_principled_node(principled_node=principled_node,
    base_color=(0.6, 0.6, 0.6, 1.0),
    metallic=0.2,
    specular=0.5,
    roughness=0.3)

links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
M.data.materials.append(mat)
M.rotation_euler = rotation

mesh = M.data
values = [True] * len(mesh.polygons)
mesh.polygons.foreach_set('use_smooth', values)

camera = D.objects['Camera']
bpy.ops.view3d.camera_to_view_selected()

R.filepath = 'render-full.png'
R.resolution_x = 1920
R.resolution_y = 1080
R.film_transparent = True

R.engine = 'CYCLES'
C.scene.cycles.samples = 256
C.scene.cycles.use_adaptive_sampling = True
C.scene.cycles.use_denoising = False

C.scene.cycles.device = 'GPU'
C.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

C.preferences.addons['cycles'].preferences.get_devices()
for d in C.preferences.addons['cycles'].preferences.devices:
    d['use'] = 1

bpy.ops.render.render(write_still=True)

M.select_set(True)
bpy.ops.object.delete()

all_materials = []
for i, base_color in enumerate(colors):
    mat = add_material(f'Material-{i}', use_nodes=True, make_node_tree_empty=True)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    set_principled_node(principled_node=principled_node,
        base_color=base_color,
        metallic=0.2,
        specular=0.5,
        roughness=0.3)

    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    all_materials.append(mat)

output = io.StringIO()
for dir in [ 'r16' ]:
    prefix = os.path.join(directory, dir)
    print()
    paths = glob.glob(prefix + '/p*.stl')

    for path in tqdm(paths):
        basename = os.path.basename(path)
        basename = basename.split('.')[0]

        with redirect_stdout(output):
            bpy.ops.import_mesh.stl(filepath=path)

        M = D.objects[basename]

        k = int(basename[1:])
        ik = k % len(colors)
        M.data.materials.append(all_materials[ik])
        M.rotation_euler = rotation

        mesh = M.data
        values = [True] * len(mesh.polygons)
        mesh.polygons.foreach_set('use_smooth', values)

    bpy.ops.object.select_all(action='DESELECT')
    for o in D.objects:
        if o.type == 'MESH':
            o.select_set(True)
        else:
            o.select_set(False)

    R.filepath = f'render-{dir}.png'
    bpy.ops.render.render(write_still=True)
    bpy.ops.object.delete()
    
from figuregen.util import image
import numpy as np
import simpleimageio

full = simpleimageio.read('render-full.png')
patched = simpleimageio.read('render-r16.png')
split = image.SplitImage([ full, patched ], vertical=True).get_image()

print(type(split), split.shape)

alpha = np.sum(full, axis=-1) > 1e-3
print(alpha.shape, np.sum(split, axis=-1))
split = np.concatenate((split, alpha[..., None]), axis=-1)
print(type(split), split.shape)

simpleimageio.write('split.png', split)