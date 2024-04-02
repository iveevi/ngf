import os
import bpy
import mathutils
import numpy as np
import sys

from typing import Tuple

C = bpy.context
D = bpy.data
R = bpy.context.scene.render
W = bpy.context.scene.world

def clean_nodes(nodes: bpy.types.Nodes) -> None:
    for node in nodes:
        nodes.remove(node)

def add_material(name: str = 'Material', use_nodes: bool = False, make_node_tree_empty: bool = False) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = use_nodes

    if use_nodes and make_node_tree_empty:
        clean_nodes(material.node_tree.nodes)

    return material

def set_principled_node(principled_node: bpy.types.Node,
                        base_color: Tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0),
                        metallic: float = 0.0,
                        specular: float = 0.5,
                        roughness: float = 0.5,
                        ior: float = 1.45,
                        transmission: float = 0.0) -> None:
    principled_node.inputs['Base Color'].default_value = base_color
    principled_node.inputs['Metallic'].default_value = metallic
    principled_node.inputs['Roughness'].default_value = roughness
    principled_node.inputs['IOR'].default_value = ior
    principled_node.inputs['Transmission Weight'].default_value = transmission

directory = os.path.dirname(__file__)

path = sys.argv[-1]
assert os.path.exists(path)

D.objects[1].select_set(True)
D.objects[2].select_set(True)
bpy.ops.object.delete()

bpy.ops.import_mesh.stl(filepath=path)
print(list(bpy.data.objects))

enode = W.node_tree.nodes.new('ShaderNodeTexEnvironment')
enode.image = bpy.data.images.load('/home/venki/downloads/rural_crossroads_2k.hdr')
# enode.image = bpy.data.images.load('/home/venki/projects/ngf/media/environment.hdr')

node_tree = W.node_tree
node_tree.links.new(enode.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])

M = D.objects[0]
mx = M.dimensions.x
my = M.dimensions.y
mz = M.dimensions.z
extent = np.max([ mx, my, mz ])
print('extent', extent)
M.scale = [ 1/extent, 1/extent, 1/extent ]

mat = add_material('Material_Left', use_nodes=True, make_node_tree_empty=True)
nodes = mat.node_tree.nodes
links = mat.node_tree.links
output_node = nodes.new(type='ShaderNodeOutputMaterial')
principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')

set_principled_node(principled_node=principled_node,
    base_color=(0.7, 0.7, 0.7, 1.0),
    metallic=0.0,
    specular=0.5,
    roughness=0.5)

links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
M.data.materials.append(mat)

bpy.ops.view3d.camera_to_view_selected()

R.filepath = os.path.join(directory, 'render.png')
R.resolution_x = 1920
R.resolution_y = 1080
R.film_transparent = True

R.engine = 'CYCLES'
C.scene.cycles.samples = 64
C.scene.cycles.use_adaptive_sampling = True
C.scene.cycles.use_denoising = False

C.scene.cycles.device = 'GPU'
C.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

C.preferences.addons['cycles'].preferences.get_devices()
for d in C.preferences.addons['cycles'].preferences.devices:
    d['use'] = 1

bpy.ops.render.render(write_still=True)
