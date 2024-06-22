import os
import bpy
import numpy as np
import sys
import glob

sys.path.append(os.path.dirname(__file__))
from common import *
from rotations import rotations

C = bpy.context
D = bpy.data
R = bpy.context.scene.render
W = bpy.context.scene.world

D.objects[1].select_set(True)
D.objects[2].select_set(True)
bpy.ops.object.delete()

# TODO: sort to use the same view

enode = W.node_tree.nodes.new('ShaderNodeTexEnvironment')
enode.image = bpy.data.images.load('media/environment.hdr')
node_tree = W.node_tree
node_tree.nodes['Background'].inputs['Strength'].default_value = 0.5
node_tree.links.new(enode.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])

# geometry_node = W.node_tree.nodes.new('ShaderNodeNewGeometry')
# rotation_node = W.node_tree.nodes.new('ShaderNodeVectorRotate')
# rotation_node.inputs['Axis'].default_value = (1, 1, -1)
# rotation_node.inputs['Angle'].default_value = np.radians(162)
# W.node_tree.links.new(geometry_node.outputs['True Normal'], rotation_node.inputs['Vector'])
# W.node_tree.links.new(rotation_node.outputs['Vector'], W.node_tree.nodes['World Output'].inputs['Surface'])

mat = add_material('Main', use_nodes=True, make_node_tree_empty=True)
nodes = mat.node_tree.nodes
links = mat.node_tree.links
output_node = nodes.new(type='ShaderNodeOutputMaterial')
principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
set_principled_node(principled_node=principled_node,
    base_color=(0.5, 0.4, 0.45, 1.0),
    metallic=0.7, roughness=0.2, coat_ior=1.0, coat_roughness=0.001)

directory = sys.argv[-1]
model = sys.argv[-2]
init = False
    
# TODO: util function to configure
R.resolution_x = 1920
R.resolution_y = 1080
R.film_transparent = True

R.engine = 'CYCLES'
# R.engine = 'BLENDER_EEVEE'
C.scene.cycles.samples = 256
C.scene.cycles.use_adaptive_sampling = True
C.scene.cycles.use_denoising = False

C.scene.cycles.device = 'GPU'
C.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

C.preferences.addons['cycles'].preferences.get_devices()
for d in C.preferences.addons['cycles'].preferences.devices:
    d['use'] = 1

destination = os.path.join(directory, 'blender')
os.makedirs(directory, exist_ok=True)

for path in glob.glob(directory + '/*.stl'):
    basename = os.path.basename(path)
    basename = basename.split('.')[0]
    bpy.ops.import_mesh.stl(filepath=path)
    print(list(bpy.data.objects))

    M = D.objects[basename]
    normalize(M)
    # if not basename.endswith('ngf'):
    #     normalize(M)
    # else:
    #     print('NORMALIZING')

    # mx = M.dimensions.x
    # my = M.dimensions.y
    # mz = M.dimensions.z
    # # extent = np.max([ mx, my, mz ])
    # extent = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)/2
    # print('extent', extent)
    # M.scale = [ 1/extent, 1/extent, 1/extent ]

    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    M.data.materials.append(mat)
    M.rotation_euler = rotations(model)
    # M.rotation_euler = rotations(basename)

    # mesh = M.data
    # values = [True] * len(mesh.polygons)
    # mesh.polygons.foreach_set('use_smooth', values)

    # if not init:
    camera = D.objects['Camera']
    bpy.ops.view3d.camera_to_view_selected()
    # init = True

    # R.filepath = 'results/blender/' + basename + '.png'
    R.filepath = os.path.join(destination, basename + '.png')
    bpy.ops.render.render(write_still=True)

    M.select_set(True)
    bpy.ops.object.delete()
    # break
