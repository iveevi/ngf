import meshio
import numpy as np
import os
import polyscope as ps
import sys
import torch

from scripts.load_xml import load_scene

assert len(sys.argv) == 3, "Usage: python preview.py <model> <scene>"

model = sys.argv[1]
scene = sys.argv[2]

model = meshio.read(model)
scene = load_scene(scene)['view_mats']

print(model)
# print(scene)

views = []
for mat in scene:
    origin = torch.tensor([0, 0, 0, 1], device='cuda', dtype=torch.float32)
    origin = mat @ origin
    views.append(origin[:3].cpu().numpy())
views = np.array(views)

print(views)

ps.init()
ps.register_surface_mesh("model", model.points, model.cells_dict['triangle'])
ps.register_point_cloud("views", views)
ps.show()
