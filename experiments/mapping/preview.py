import numpy as np
import polyscope as ps
import sys
import torch
import optext
import meshio

sys.path.append('../../source')
from ngf import load_ngf

ngf = torch.load('experimental.pt')
ngf = load_ngf(ngf)

def normalize(v):
    min = v.min(axis=0)
    max = v.max(axis=0)
    center = (min + max) / 2
    scale = np.sqrt(np.square(max - min).sum()) / 2
    return (v - center) / scale

base = meshio.read('base.ply')
base_v = normalize(base.points)
base_f = base.cells_dict['triangle']
s = base.point_data['s']
t = base.point_data['t']
base_uv = np.stack((s, t), axis=-1)

ref = meshio.read('ref.ply')
ref_v = normalize(ref.points)
ref_f = ref.cells_dict['triangle']
s = ref.point_data['s']
t = ref.point_data['t']
ref_uv = np.stack((s, t), axis=-1)

rate = 32
sample = ngf.sample_uniform(rate)

ST = torch.from_numpy(base_uv).cuda()
BV = torch.from_numpy(base_v).cuda()

# Get reindexed indices
quads = []
for c in ngf.complexes.cpu():
    q = []
    for i in c:
        v = ngf.points[i].detach().cpu().numpy()
        j = np.argmin(np.linalg.norm(v - base_v, axis=-1))
        q.append(j)

    quads.append(q)

Q = np.array(quads)
Q = torch.from_numpy(Q).cuda()

P, U, V = sample
C = ngf.complexes[P, :]
C = Q[P, :]
Up, Um = U.unsqueeze(-1), (1.0 - U).unsqueeze(-1)
Vp, Vm = V.unsqueeze(-1), (1.0 - V).unsqueeze(-1)

luv00 = ST[C[:, 0]] * Um * Vm
luv01 = ST[C[:, 1]] * Up * Vm
luv10 = ST[C[:, 3]] * Um * Vp
luv11 = ST[C[:, 2]] * Up * Vp
luv = luv00 + luv01 + luv10 + luv11

# print('complex', C.shape, ngf.points.shape, lp.shape)

V = ngf.eval(*sample).detach()
F = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)

ngf_v, ngf_f, ngf_uv = V.cpu().numpy(), F.cpu().numpy(), luv.cpu().numpy()

print('shapes', ngf_v.shape, ngf_uv.shape, ngf_f.shape)

ps.init()
ps.set_ground_plane_mode('shadow_only')

m = ps.register_surface_mesh('ref', ref_v, ref_f)
m.add_parameterization_quantity('uv', ref_uv, enabled=True)

m = ps.register_surface_mesh('base', base_v, base_f)
m.add_parameterization_quantity('uv', base_uv, enabled=True)

m = ps.register_surface_mesh('ngf', ngf_v, ngf_f)
m.add_parameterization_quantity('uv', ngf_uv, enabled=True)

# m = ps.register_surface_mesh('ngf base', base_v, Q.cpu().numpy())

ps.show()
