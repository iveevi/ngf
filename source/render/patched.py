import os
import bpy
import numpy as np
import sys
import glob
import io
import os
import torch
import trimesh
import ngfutil
import shutil

# Blender
C = bpy.context
D = bpy.data
R = bpy.context.scene.render
W = bpy.context.scene.world

current = os.path.dirname(__file__)
upper = os.path.join(current, os.path.pardir)
sys.path += [current, upper]

from common import *
from ngf import NGF
from util import make_cmap

# Setting up
D.objects[0].select_set(True)
D.objects[1].select_set(True)
D.objects[2].select_set(True)
bpy.ops.object.delete()

node_tree = W.node_tree
enode = W.node_tree.nodes.new('ShaderNodeTexEnvironment')
enode.image = bpy.data.images.load('media/environment.hdr')
node_tree.links.new(enode.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])

# Reference mesh
rotation = np.radians([ 90, 0, 0 ])

R.resolution_x = 1920
R.resolution_y = 1080

R.engine = 'CYCLES'
C.scene.cycles.samples = 256
C.scene.cycles.use_adaptive_sampling = True
C.scene.cycles.use_denoising = False

C.scene.cycles.device = 'GPU'

C.preferences.addons['cycles'].preferences.get_devices()
for d in C.preferences.addons['cycles'].preferences.devices:
    d['use'] = 1

# Neural geometry fields
paths = [
    # 'results/torched/nefertiti-lod2500-f20.pt',
    # 'results/torched/armadillo-lod2500-f20.pt',
    # 'results/torched/buddha-lod2500-f20.pt',
    # 'results/torched/dragon-lod2500-f20.pt',
    # 'results/torched/lucy-lod2500-f20.pt',
    # 'results/torched/xyz-lod2500-f20.pt',
    'results/torched/ganesha-lod2500-f20.pt',
    # 'results/torched/metratron-lod2500-f20.pt',
    # 'results/torched/einstein-lod2500-f20.pt',
]

for path in paths:
    name = os.path.basename(path)
    name = name.split('.')[0]

    ngf = NGF.from_pt(path)

    directory = 'results/viz/patched'
    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.makedirs(directory, exist_ok=True)

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

    # for rate in [ 2, 4, 8, 16 ]:
    for rate in [ 16 ]:
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

    from tqdm import tqdm

    all_materials = []
    for i, base_color in enumerate(colors):
        mat = add_material(f'Material-{i}', use_nodes=True, make_node_tree_empty=True)
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        set_principled_node(principled_node=principled_node,
            base_color=base_color, metallic=0, roughness=0.4)

        links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
        all_materials.append(mat)

    for dir in [ 'r16' ]:
        prefix = os.path.join(directory, dir)
        paths = glob.glob(prefix + '/p*.stl')

        for path in paths:
            basename = os.path.basename(path)
            basename = basename.split('.')[0]

            bpy.ops.import_mesh.stl(filepath=path)

            M = D.objects[basename]

            k = int(basename[1:])
            ik = k % len(colors)
            M.data.materials.append(all_materials[ik])
            M.rotation_euler = rotation
            # M.rotation_euler.z = '#frame/25'

            mesh = M.data
            values = [True] * len(mesh.polygons)
            mesh.polygons.foreach_set('use_smooth', values)

        # Join all the patches into one
        patches = [obj for obj in bpy.context.scene.objects if obj.name.startswith('p')]

        bpy.context.view_layer.objects.active = patches[0]
        for obj in patches:
            obj.select_set(True)

        bpy.ops.object.join()

        bpy.context.active_object.name = name
