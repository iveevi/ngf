import meshio
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util import *
from configurations import *

# Arguments
file   = '../results/mask/full.pt'
source = 'parametrized.obj'

assert os.path.exists(file), f'file does not exist: {file}'
assert os.path.exists(source), f'file does not exist: {source}'

# Load the source object
mesh = meshio.read(source)
# V = torch.tensor(mesh.points).float().cuda()
# C = torch.tensor(mesh.cells_dict['quad']).int().cuda()
# UV = torch.tensor(mesh.point_data['uv']).float().cuda()

def shorted_quads(V, C, sample_rate=16):
    quads = []
    for c in range(C.shape[0]):
        offset = c * sample_rate * sample_rate
        for i in range(sample_rate - 1):
            for j in range(sample_rate - 1):
                a = offset + i * sample_rate + j
                c = offset + (i + 1) * sample_rate + j
                b, d = a + 1, c + 1
                quads.append([a, b, d, c])

    return np.array(quads)

# Recursively find all simple meshes and model configurations
print('Loading NSC representation: %s' % file)

data = torch.load(file)

m = data['model']
c = data['complexes']
p = data['points'].detach()
f = data['features'].detach()

mesh = meshio.Mesh(points=p.cpu().numpy(), cells={'quad': c.cpu().numpy()})
print('Writing source object')
meshio.write('source.obj', mesh)
print(mesh)

ker = data['kernel']
ker = lerps[ker]

# Compute byte size of the representation
feature_bytes = f.numel() * f.element_size()
index_bytes = c.numel() * c.element_size()
model_bytes = sum([ p.numel() * p.element_size() for p in m.parameters() ])
vertex_bytes = p.numel() * p.element_size()
total = feature_bytes + index_bytes + model_bytes + vertex_bytes

rate = 16
lerped_points, lerped_features = sample(c, p, f, rate, kernel=ker)

vertices = m(points=lerped_points, features=lerped_features).detach()

I = shorted_indices(vertices.cpu().numpy(), c, rate)
I = torch.from_numpy(I).int()

# Compute the quasi conformality of each triangle in the patch
F = I.reshape(c.shape[0], 2 * (rate - 1) ** 2, 3)
print('Computing quasi conformality')

# First assign UV coordinates to each vertex
uv = torch.zeros((rate ** 2, 2))
for i in range(rate):
    for j in range(rate):
        uv[i * rate + j] = torch.tensor([i, j])/(rate - 1)

uv = uv.unsqueeze(0).repeat(c.shape[0], 1, 1)
uv = uv.reshape(c.shape[0] * rate ** 2, 2).cuda()

def eigs(JtJ):
    # Using quadratic formula
    a = 1
    b = -JtJ.trace()
    c = torch.det(JtJ)

    d = b ** 2 - 4 * a * c

    if d < 0:
        return torch.zeros(2)
    else:
        return torch.tensor([(-b + torch.sqrt(d)) / (2 * a), (-b - torch.sqrt(d)) / (2 * a)])

from tqdm import tqdm

qcs = torch.zeros(I.shape[0]).cuda()

bar = tqdm(total=F.shape[0])
for c, f in enumerate(F):
    local_vertices = vertices[f].reshape(-1, 3, 3)
    local_uv = uv[f].reshape(-1, 3, 2)
    # print(local_uv)
    # print(f.shape, local_vertices.shape, local_uv.shape)

    # TODO: kernel...
    for i, (vs, uvs) in enumerate(zip(local_vertices, local_uv)):
        center = vs.mean(0)
        center_uv = uvs.mean(0)

        du = (center - vs[0]) / (center_uv[0] - uvs[0, 0])
        dv = (center - vs[0]) / (center_uv[1] - uvs[0, 1])

        # print(center, center_uv)
        # print(vs, uvs)

        # Approximate dvertex/du and dvertex/dv with the central difference
        # du = 0
        # dv = 0
        #
        # for j in range(3):
        #     du += (center - vs[j]) / (center_uv[0] - uv[j, 0])
        #     dv += (center - vs[j]) / (center_uv[1] - uv[j, 1])
        #
        # du /= 3
        # dv /= 3

        J = torch.stack([du, dv], dim=1)
        Mt = J.t().matmul(J)
        qc = (0.5 * Mt.trace()) ** 0.5

        assert qc.is_nonzero()
        # assert not torch.isnan(qc) and not torch.isinf(qc)

        index = c * (rate - 1) ** 2 + i
        qcs[index] = qc

    bar.update(1)
bar.close()

print('# of zeros:', (qcs == 0).sum().item(), 'total:', qcs.numel())

# TODO: area weighted average
qcs_mean = qcs.mean()

triangles = vertices[I].reshape(-1, 3, 3)
areas = triangles[:, 0].cross(triangles[:, 1]).norm(dim=1) / 2
qcs_area = qcs * areas
qcs_weighted_mean = qcs_area.sum() / areas.sum()

print('average qc:', qcs_mean.item())
print('weighted average qc:', qcs_weighted_mean.item())

# TODO: parametrize the complexes, then inteprolate for uvs of the actual mesh...

# TODO: convert to per vertex map...
qcs_vertex = torch.zeros(vertices.shape[0]).cuda()
weights = torch.zeros(vertices.shape[0]).cuda()
for i, (f, qc) in enumerate(zip(F, qcs)):
    for vi in f:
        qcs_vertex[vi] += qc
        weights[vi] += 1

qcs_vertex /= weights

import polyscope as ps

ps.init()
ps.set_ground_plane_mode('none')
m = ps.register_surface_mesh('mesh', vertices.cpu().numpy(), I.cpu().numpy())
m.add_scalar_quantity('qc', qcs_vertex.cpu().numpy(), enabled=True, cmap='coolwarm')
m.add_scalar_quantity('qc face', qcs.cpu().numpy(), enabled=True, defined_on='faces', cmap='coolwarm')
m.set_smooth_shade(True)
ps.show()
