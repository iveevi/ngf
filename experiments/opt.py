import argparse
import meshio
import numpy as np
import os
import torch

from tqdm import trange

# from largesteps.parameterize import from_differential, to_differential
# from largesteps.geometry import compute_matrix, laplacian_uniform
# from largesteps.optimize import AdamUniform

from torch.utils.cpp_extension import load

from mlp import *
from util import *

# TODO: apply additional inverse rendering to refine even further...

# Arguments
parser = argparse.ArgumentParser(description='neural subdivision complexes: mutliple complexes')
parser.add_argument('--target', type=str, help='target geometry')
parser.add_argument('--source', type=str, help='source geometry')
parser.add_argument('--resolution', type=int, default=16, help='target resolution of each complex')
parser.add_argument('--output', type=str, default='', help='tensor output file')

args = parser.parse_args()
if args.output == '':
    args.output = os.path.splitext(args.source)[0] + '_opt.pt'

# Load all necessary extensions
if not os.path.exists('../build'):
    os.makedirs('../build')

geom = load(name='geom',
        sources=[ '../ext/geometry.cpp' ],
        extra_include_paths=[ '../glm' ],
        build_directory='../build')

print('Loaded geometry extension')

# TODO: make geom obselete; use a new file instead

casdf = load(name='casdf',
        sources=[ '../ext/casdf.cu' ],
        extra_include_paths=[ '../glm' ],
        build_directory='../build')

print('Loaded casdf extension')

# Load the target object
target = meshio.read(args.target)
source = meshio.read(args.source)

print('Loaded target mesh: %s' % args.target)
print('Loaded source mesh: %s' % args.source)

tch_target_vertices = torch.from_numpy(target.points).float().cuda()
tch_target_triangles = torch.from_numpy(target.cells_dict['triangle']).int().cuda()

# TODO: automatically compute normals if not provided...
cas = casdf.geometry(tch_target_vertices.cpu(), tch_target_vertices.cpu(), tch_target_triangles.cpu())
cas = casdf.cas_grid(cas, 32)

# points = source.points
points = torch.from_numpy(source.points).float().cuda()
complexes = torch.from_numpy(source.cells_dict['quad']).int().cuda()
features = torch.randn((points.shape[0], POINT_ENCODING_SIZE), requires_grad=True, device='cuda', dtype=torch.float32)

print('complexes:', complexes.shape, type(complexes), complexes.device)
print('points:', points.shape, type(points), points.device)
print('encodings:', features.shape, type(features), features.device)

distances = torch.cdist(features, features)
print('distance max = %f' % distances.max())

# TODO: geom function?
def make_cmap(complexes, LP, sample_rate):
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

def sample_surface(V, T, count=1000):
    # Sample random triangles and baricentric coordinates
    rT = torch.randint(0, T.shape[0], (count,)).cuda()
    rB = torch.rand((count, 2)).cuda()
    rBu, rBv = rB[:, 0], rB[:, 1]
    rBu_sqrt = rBu.sqrt()
    w0 = 1.0 - rBu_sqrt
    w1 = rBu_sqrt * (1.0 - rBv)
    w2 = rBu_sqrt * rBv
    rB = torch.stack([w0, w1, w2], dim=-1)
    return V[T[rT, 0]] * rB[:, 0].unsqueeze(-1) + \
            V[T[rT, 1]] * rB[:, 1].unsqueeze(-1) + \
            V[T[rT, 2]] * rB[:, 2].unsqueeze(-1)

sample_rate = args.resolution
LP, LE, UV = sample(complexes, points, features, sample_rate)
I = indices(complexes, sample_rate=sample_rate)

history = {}

# cas.precache_query_device(LP)
Tv          = LP.clone().requires_grad_(True)
I_tch       = torch.from_numpy(I).int().cuda()

cmap        = make_cmap(complexes, LP, sample_rate)
F, remap    = geom.sdc_weld(complexes.cpu(), cmap, Tv.shape[0], sample_rate)
F           = F.cuda()
vgraph      = casdf.vertex_graph(F.cpu())

Tv_opt      = torch.optim.Adam([Tv], lr=0.01)

closest     = torch.zeros((Tv.shape[0], 3)).cuda()
bary        = torch.zeros((Tv.shape[0], 3)).cuda()
dist        = torch.zeros(Tv.shape[0]).cuda()
index       = torch.zeros(Tv.shape[0], dtype=torch.int32).cuda()

samples     = 10_000
sample_bary = torch.zeros((samples, 3), dtype=torch.float32).cuda()
sample_tris = torch.zeros(samples, dtype=torch.int32).cuda()

history = {}
for i in trange(1_000):
    # Direct loss computation
    rate = cas.precache_query_device(Tv)

    cas.precache_device()
    cas.query_device(Tv, closest, bary, dist, index)

    direct_loss = torch.sum((closest - Tv).square())

    # Sampled loss computation
    Vrandom = sample_surface(tch_target_vertices, tch_target_triangles, count=samples)
    casdf.barycentric_closest_points(Tv, F, Vrandom, sample_bary, sample_tris)
    Ts = F[sample_tris]
    Vreconstructed = Tv[Ts[:, 0]] * sample_bary[:, 0].unsqueeze(-1) + \
                        Tv[Ts[:, 1]] * sample_bary[:, 1].unsqueeze(-1) + \
                        Tv[Ts[:, 2]] * sample_bary[:, 2].unsqueeze(-1)

    sampled_loss = torch.sum((Vrandom - Vreconstructed).square())

    # TODO: tirangle area min/maxing...
    # print('direct = %f, loss = %f' % (direct_loss, sampled_loss))
    loss = direct_loss + sampled_loss

    history.setdefault('direct', []).append(direct_loss.item())
    history.setdefault('sampled', []).append(sampled_loss.item())

    Tv_opt.zero_grad()
    loss.backward()
    Tv_opt.step()

    # TODO: stop at a certain point...
    if i > 0 and i < 750 and i % 100 == 0:
        with torch.no_grad():
            V = vgraph.smooth_device(Tv, 1.0)
            Tv.data.copy_(V)

# Save this tensor
Tv = geom.sdc_separate(Tv.detach().cpu(), remap).cuda()
torch.save(Tv, args.output)
print('Saved tensor to', args.output)

# TODO: plot all the losses...

import polyscope as ps

C_colors = color_code_complexes(complexes, sample_rate)
Tv = Tv.detach().cpu()
sI = shorted_indices(Tv, complexes, sample_rate)

ps.init()
ps.register_surface_mesh('target', target.points, target.cells_dict['triangle'])
ps.register_surface_mesh('Tv', Tv, sI).add_color_quantity('complex', C_colors, defined_on='faces')
ps.set_ground_plane_mode('none')
ps.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

# TODO: figure(filename) in util
plt.plot(history['direct'], label='direct')
plt.plot(history['sampled'], label='sampled')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.savefig('optimization.png')
