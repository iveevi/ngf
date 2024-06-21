import argparse
import glob
import sys
import os
import re
import bpy
import numpy as np

from mathutils import Color

marker = '--'
marker = sys.argv.index(marker)
assert marker > 0

argv = sys.argv[marker + 1:]
print(argv)

signature = argv[0]
begin = int(argv[1])
stride = int(argv[2])
frames = argv[3]

signature = signature.replace('@', '*')
print(signature, begin, stride)

ordered = {}

for file in glob.glob(signature):
    directory = os.path.dirname(file)
    file = os.path.basename(file)
    n = re.search(r'\d+', file)
    n = int(n.group(0))
    ordered[n] = os.path.join(directory, file)

limit = max(ordered.keys())
print('limit', limit)

# Render the desired things
C = bpy.context
D = bpy.data
R = bpy.context.scene.render
W = bpy.context.scene.world

D.objects[1].select_set(True)
D.objects[2].select_set(True)
bpy.ops.object.delete()

# World shading
T = W.node_tree

ray = T.nodes.new('ShaderNodeLightPath')
rgb = T.nodes.new('ShaderNodeRGB')
mixer = T.nodes.new('ShaderNodeMixShader')
enode = T.nodes.new('ShaderNodeTexEnvironment')

background = T.nodes['Background']
world = T.nodes['World Output']

rgb.outputs[0].default_value = (1, 1, 1, 1)
background.inputs['Strength'].default_value = 1.5
enode.image = bpy.data.images.load('media/environment.hdr')

T.links.new(enode.outputs['Color'], background.inputs['Color'])
T.links.new(mixer.inputs['Fac'], ray.outputs['Is Camera Ray'])
T.links.new(mixer.inputs[1], background.outputs['Background'])
T.links.new(mixer.inputs[2], rgb.outputs[0])
T.links.new(mixer.outputs[0], world.inputs['Surface'])

C.scene.view_settings.view_transform = 'Standard'

# Rendering configuration
R.resolution_x = 1920
R.resolution_y = 1080

R.engine = 'CYCLES'
C.scene.cycles.samples = 256
C.scene.cycles.use_adaptive_sampling = True
C.scene.cycles.use_denoising = False

C.scene.cycles.device = 'GPU'
C.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

C.preferences.addons['cycles'].preferences.get_devices()
for d in C.preferences.addons['cycles'].preferences.devices:
    d['use'] = 1

camera = D.objects[0]
camera.rotation_euler.x = np.radians(-90)
camera.rotation_euler.y = np.radians(180)
camera.rotation_euler.z = 0
camera.location.x = 0
camera.location.y = 10
camera.location.z = 0

for f, i in enumerate(range(begin, limit + 1, stride)):
    file = ordered[i]
    print('i', i, '->', file)
    bpy.ops.import_mesh.stl(filepath=file)
    R.filepath = os.path.join(frames, f'f{f:04d}.png')

    basename = os.path.basename(file)
    basename = basename.split('.')[0]
    M = D.objects[basename]

    M.location.x = 0
    M.location.y = -2
    M.location.z = 0

    M.rotation_euler.x = np.radians(-90)
    M.rotation_euler.y = 0
    M.rotation_euler.z = 0

    M.scale.x = 2
    M.scale.y = 2
    M.scale.z = 2

    # bpy.ops.view3d.camera_to_view_selected()
    bpy.ops.render.render(write_still=True)

    M.select_set(True)
    bpy.ops.object.delete()
