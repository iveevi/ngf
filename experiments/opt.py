import argparse
import imageio.v2 as imageio
import meshio
import numpy as np
import os
import torch

from tqdm import trange

from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix, laplacian_uniform
from largesteps.optimize import AdamUniform

from torch.utils.cpp_extension import load

from mlp import *
from util import *
from render import *


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

# Load the target object
target = meshio.read(args.target)
source = meshio.read(args.source)

print('Loaded target mesh: %s' % args.target)
print('Loaded source mesh: %s' % args.source)

tch_target_vertices = torch.from_numpy(target.points).float().cuda()
tch_target_triangles = torch.from_numpy(target.cells_dict['triangle']).int().cuda()

tch_target_face_normals = compute_face_normals(tch_target_vertices, tch_target_triangles)
tch_target_vertex_normals = compute_vertex_normals(tch_target_vertices, tch_target_triangles, tch_target_face_normals)

# Generate random camera views
# TODO: provide a selection of environment maps?
# and pass a config file for each mesh instead of command line arguments?
# which also includes the netx stage paramters (for training)

center = torch.mean(tch_target_vertices, dim=0)

environment = torch.tensor(imageio.imread('environment.hdr', format='HDR-FI'), device='cuda')
alpha = torch.ones((*environment.shape[:2], 1), device='cuda')
environment = torch.cat((environment, alpha), dim=-1)

def lookat(eye, center, up):
    normalize = lambda x: x / torch.norm(x)

    f = normalize(eye - center)
    u = normalize(up)
    s = normalize(torch.cross(f, u))
    u = torch.cross(s, f)

    dot_f = torch.dot(f, eye)
    dot_u = torch.dot(u, eye)
    dot_s = torch.dot(s, eye)

    return torch.tensor([
        [s[0], u[0], -f[0], -dot_s],
        [s[1], u[1], -f[1], -dot_u],
        [s[2], u[2], -f[2], dot_f],
        [0, 0, 0, 1]
    ], device='cuda', dtype=torch.float32)

views = []
for i in range(10):
    up = torch.tensor([0.0, -1.0, 0.0], device='cuda')
    r = torch.randn(3, device='cuda') * 0.5
    phi, theta = (r[0] * np.pi).item(), (r[1] * np.pi).item()
    radius = (r[2] * 5.0 + 1.0).item()
    eye = torch.tensor([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], device='cuda', dtype=torch.float32) * radius + center
    views.append(lookat(eye, center, up))

# TODO: use lookat?

print('Generated %d views' % len(views))

renderer = NVDRenderer(views, environment)
print('backgrounds:', renderer.bgs.shape)

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

# TODO: automatically compute normals if not provided...
cas = casdf.geometry(tch_target_vertices.cpu(), tch_target_triangles.cpu())
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

Tv_opt      = torch.optim.Adam([Tv], lr=1e-3)

closest     = torch.zeros((Tv.shape[0], 3)).cuda()
bary        = torch.zeros((Tv.shape[0], 3)).cuda()
dist        = torch.zeros(Tv.shape[0]).cuda()
index       = torch.zeros(Tv.shape[0], dtype=torch.int32).cuda()

samples     = 10_000
sample_bary = torch.zeros((samples, 3), dtype=torch.float32).cuda()
sample_tris = torch.zeros(samples, dtype=torch.int32).cuda()

# Tv_face_normals = compute_face_normals(Tv, F)
# print('nans:', torch.isnan(Tv).sum().item(), torch.isnan(Tv_face_normals).sum().item())
# Tv_vertex_normals = compute_vertex_normals(Tv, F, Tv_face_normals)
# print('nans:', torch.isnan(Tv).sum().item(), torch.isnan(Tv_vertex_normals).sum().item())

target_imgs = renderer.render(tch_target_vertices, tch_target_vertex_normals, tch_target_triangles)

# for i in range(len(imgs)):
#     plt.subplot(2, len(imgs), i + 1)
#     plt.imshow(imgs[i].cpu().numpy())
#     plt.axis('off')
#
# imgs = renderer.render(Tv, Tv_vertex_normals, F)
# imgs = torch.clamp(imgs, 0.0, 1.0)
# for i in range(len(imgs)):
#     plt.subplot(2, len(imgs), len(imgs) + i + 1)
#     plt.imshow(imgs[i].detach().cpu().numpy())
#     plt.axis('off')
#
# plt.show()

M = compute_matrix(Tv.detach(), F, 0.001)
U = to_differential(M, Tv.detach())
print('Tv initial shape:', Tv.shape)

# U.requires_grad = True
# Tv_opt = AdamUniform([ U ], 1e-3)

# TODO: use the UV parametrization

history = {}
for i in trange(1_000):
    # Tv = from_differential(M, U, 'Cholesky')

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

    # Laplacian loss
    Tv_smoothed = vgraph.smooth_device(Tv, 1.0)
    laplacian_loss = torch.sum((Tv - Tv_smoothed).square())

    # TODO: match normal vectors?

    # Rendering loss
    # Tv_Fn = compute_face_normals(Tv, I_tch)
    # Tv_Vn = compute_vertex_normals(Tv, I_tch, Tv_Fn)

    # imgs = renderer.render(Tv, Tv_Vn, F)
    # rendered_loss = 1e1 * torch.mean((imgs - target_imgs).abs())
    # print('rendered loss:', rendered_loss.item(), rendered_loss)

    # TODO: tirangle area min/maxing...
    # print('direct:', direct_loss.item(), 'sampled:', sampled_loss.item(), 'laplacian:', laplacian_loss.item(), 'rendered:', rendered_loss.item())
    loss = direct_loss + sampled_loss + laplacian_loss # + rendered_loss

    history.setdefault('direct', []).append(direct_loss.item())
    history.setdefault('sampled', []).append(sampled_loss.item())
    history.setdefault('laplacian', []).append(laplacian_loss.item())
    # history.setdefault('rendered', []).append(rendered_loss.item())

    Tv_opt.zero_grad()
    loss.backward()
    Tv_opt.step()

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
ps.register_surface_mesh('Tv init', LP.cpu(), sI).add_color_quantity('complex', C_colors, defined_on='faces')
ps.set_ground_plane_mode('none')
ps.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

# TODO: figure(filename) in util
plt.plot(history['direct'], label='direct')
plt.plot(history['sampled'], label='sampled')
plt.plot(history['laplacian'], label='laplacian')
# plt.plot(history['rendered'], label='rendered')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.savefig('optimization.png')
