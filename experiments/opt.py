import argparse
import imageio.v2 as imageio
import meshio
import numpy as np
import os
import sys
import torch

from tqdm import trange

from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix, laplacian_uniform
from largesteps.optimize import AdamUniform

from torch.utils.cpp_extension import load

sys.path.append('..')
from scripts.load_xml import load_scene

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

scene = load_scene('../scenes/nefertiti/nefertiti.xml')
# print('Loaded scene:', scene)

environment = scene['envmap']
# environment = 0.1 * torch.tensor(imageio.imread('environment.hdr', format='HDR-FI'), device='cuda')
# alpha = torch.ones((*environment.shape[:2], 1), device='cuda')
# environment = torch.cat((environment, alpha), dim=-1)

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

radius = 4.0
splitsTheta = 10
splitsPhi = 10

for theta in np.linspace(0, 2 * np.pi, splitsTheta, endpoint=False):
    for i, phi in enumerate(np.linspace(np.pi * 0.05, np.pi * 0.95, splitsPhi, endpoint=True)):
        # Spiral as a function of phi
        ptheta = (2 * np.pi / splitsTheta) * i/splitsPhi
        theta += ptheta

        eye_offset = torch.tensor([
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(phi),
            radius * np.cos(theta) * np.sin(phi)
        ], device='cuda', dtype=torch.float32)

        # TODO: random perturbations to the angles

        normalized_eye_offset = eye_offset / torch.norm(eye_offset)

        canonical_up = torch.tensor([0.0, 0.0, 1.0], device='cuda')
        right = torch.cross(normalized_eye_offset, canonical_up)
        up = torch.cross(right, normalized_eye_offset)

        view = lookat(center + eye_offset, center, canonical_up)
        views.append(view)

# TODO: use lookat?

print('Generated %d views' % len(views))

# renderer = NVDRenderer(views, environment)
renderer = NVDRenderer(scene['view_mats'], environment)
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

import matplotlib.pyplot as plt

def view_images(images, splitsTheta, splitsPhi):
    plt.figure(figsize=(splitsTheta * 2, splitsPhi * 2))
    for i in range(splitsTheta * splitsPhi):
        plt.subplot(splitsPhi, splitsTheta, i + 1)
        plt.imshow(images[i].cpu())
        plt.axis('off')
    # plt.show()

M = compute_matrix(Tv.detach(), F, 200.0)
U = to_differential(M, Tv.detach())
print('Tv initial shape:', Tv.shape)

U.requires_grad = True
Tv_opt = AdamUniform([ U ], 1e-2)

# TODO: use the UV parametrization
# TODO: Subdivide...

history = {}

views = scene['view_mats']
batch = 25

intervals = []
i = 0
while i < len(views):
    start = i
    end = min(i + batch, len(views))
    intervals.append((start, end))
    i = end

print('intervals:', intervals)

for i in trange(1_000):
    # Batch the views
    # batch_indices = torch.randint(0, len(views), (batch,)).cuda()
    # print('batch indices:', batch_indices)

    for I in intervals:
        batch_indices = torch.arange(I[0], I[1]).cuda()
        Tv = from_differential(M, U, 'Cholesky')

        # Laplacian loss
        Tv_smoothed = vgraph.smooth_device(Tv, 1.0)
        laplacian_loss = torch.sum((Tv - Tv_smoothed).square())

        # TODO: match normal vectors?

        Tv_Fn = compute_face_normals(Tv, F)
        Tv_Vn = compute_vertex_normals(Tv, F, Tv_Fn)

        target_imgs = renderer.render(tch_target_vertices, tch_target_vertex_normals, tch_target_triangles, batch_indices)
        imgs = renderer.render(Tv, -Tv_Vn, F, batch_indices)
        rendered_loss = (imgs - target_imgs).abs().mean()
        print('rendered loss:', rendered_loss.item())

        # TODO: tirangle area min/maxing...
        # print('direct:', direct_loss.item(), 'sampled:', sampled_loss.item(), 'laplacian:', laplacian_loss.item(), 'rendered:', rendered_loss.item())
        # loss = direct_loss + sampled_loss + laplacian_loss # + rendered_loss
        loss = rendered_loss

        history.setdefault('laplacian', []).append(laplacian_loss.item())
        history.setdefault('rendered', []).append(rendered_loss.item())

        Tv_opt.zero_grad()
        loss.backward()
        Tv_opt.step()

view_images(target_imgs, 4, 5)
view_images(imgs.detach(), 4, 5)
plt.show()

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
plt.plot(history['laplacian'], label='laplacian')
plt.plot(history['rendered'], label='rendered')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.savefig('optimization.png')
