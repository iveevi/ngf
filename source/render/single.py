import os
import bpy
import numpy as np
import sys

sys.path.append(os.path.dirname(__file__))
from util import *
from rotations import rotations

C = bpy.context
D = bpy.data
R = bpy.context.scene.render
W = bpy.context.scene.world

D.objects[1].select_set(True)
D.objects[2].select_set(True)
bpy.ops.object.delete()

os.makedirs('results/blender', exist_ok=True)

enode = W.node_tree.nodes.new('ShaderNodeTexEnvironment')
enode.image = bpy.data.images.load('/home/venki/downloads/rural_crossroads_2k.hdr')
node_tree = W.node_tree
node_tree.links.new(enode.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])

mat = add_material('Main', use_nodes=True, make_node_tree_empty=True)
nodes = mat.node_tree.nodes
links = mat.node_tree.links
output_node = nodes.new(type='ShaderNodeOutputMaterial')
principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
set_principled_node(principled_node=principled_node,
    base_color=(0.8, 0.6, 0.35, 1.0),
    metallic=0.2, roughness=0.3)

path = sys.argv[-1]
basename = os.path.basename(path)
basename = basename.split('.')[0]

bpy.ops.import_mesh.stl(filepath=path)
print(list(bpy.data.objects))

M = D.objects[basename]
mx = M.dimensions.x
my = M.dimensions.y
mz = M.dimensions.z
# extent = np.max([ mx, my, mz ])
extent = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)/2
print('extent', extent)
M.scale = [ 1/extent, 1/extent, 1/extent ]

links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
M.data.materials.append(mat)
# M.rotation_euler = np.radians([ 90, 0, 0 ])
M.rotation_euler = rotations(basename)

mesh = M.data
values = [True] * len(mesh.polygons)
mesh.polygons.foreach_set('use_smooth', values)

camera = D.objects['Camera']
bpy.ops.view3d.camera_to_view_selected()

R.filepath = 'blender-render.png'
R.resolution_x = 1920
R.resolution_y = 1080
R.film_transparent = True

R.engine = 'CYCLES'
C.scene.cycles.samples = 16
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
