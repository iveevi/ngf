import argparse
import os
import seaborn as sns
import sys
import torch

from torch.utils.cpp_extension import load
from tqdm import trange

sns.set()

parser = argparse.ArgumentParser(description='Train a neural subdivision complex on a mesh')
parser.add_argument('--target', type=str, help='Target mesh')
parser.add_argument('--source', type=str, help='Source mesh')

args = parser.parse_args()
assert args.target is not None
assert args.source is not None

if not os.path.exists('build'):
    os.makedirs('build')

print('Loading geometry library...')
geom_cpp = load(name='geom_cpp',
        sources=[ '../ext/geometry.cpp' ],
        extra_include_paths=[ '../glm' ],
        build_directory='build',
)

print('Loading casdf library...')
casdf_cpp = load(name='casdf_cpp',
        sources=[ '../ext/casdf.cu' ],
        extra_include_paths=[ '../glm' ],
        build_directory='build',
)

print('Done loading libraries.')

import meshio
import imageio

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.geometry import compute_vertex_normals, compute_face_normals

# mesh = meshio.read('meshes/nefertiti/target.obj')
mesh = meshio.read(args.target)

environment = imageio.imread('environment.hdr', format='HDR-FI')
environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
alpha = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
environment = torch.cat((environment, alpha), dim=-1)

# scene_parameters = load_scene(filename)
scene_parameters = {}
scene_parameters['res_x'] = 1024
scene_parameters['res_y'] = 640
scene_parameters['fov'] = 45.0
scene_parameters['near_clip'] = 0.1
scene_parameters['far_clip'] = 1000.0
scene_parameters['envmap'] = environment
scene_parameters['envmap_scale'] = 1.0

#v_ref = scene_parameters['mesh-target']['vertices']
#n_ref = scene_parameters['mesh-target']['normals']
#f_ref = scene_parameters['mesh-target']['faces']

v_ref = torch.from_numpy(mesh.points).float().cuda()
f_ref = torch.from_numpy(mesh.cells_dict['triangle']).int().cuda()

fn_ref = compute_face_normals(v_ref, f_ref)
n_ref = compute_vertex_normals(v_ref, f_ref, fn_ref)

print('vertices:', v_ref.shape, 'faces:', f_ref.shape)

# Generate camera views
import numpy as np

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

radius = 5.0
splitsTheta = 3
splitsPhi = 3

center = v_ref.mean(axis=0)
for theta in np.linspace(0, 2 * np.pi, splitsTheta, endpoint=False):
    for i, phi in enumerate(np.linspace(np.pi * 0.1, np.pi * 0.9, splitsPhi, endpoint=True)):
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

        canonical_up = torch.tensor([0.0, 1.0, 0.0], device='cuda')
        right = torch.cross(normalized_eye_offset, canonical_up)
        up = torch.cross(right, normalized_eye_offset)

        view = lookat(center + eye_offset, center, canonical_up)
        views.append(view)

# TODO: use lookat?

print('Generated %d views' % len(views))
scene_parameters['view_mats'] = views

import meshio

# mesh = meshio.read('meshes/nefertiti/source.obj')
mesh = meshio.read(args.source)
assert 'quad' in mesh.cells_dict

v = mesh.points
f = mesh.cells_dict['quad']

#v = torch.from_numpy(v).float().cuda()
#print('Quadrangulated shape; vertices:', v.shape, 'faces:', f.shape)

# Configure neural subdivision complex parameters
# from models import *
from mlp import *

m = MLP_Positional_Encoding().cuda()

points = torch.from_numpy(v).float().cuda()
complexes = torch.from_numpy(f).int().cuda()
encodings = torch.zeros((points.shape[0], POINT_ENCODING_SIZE), requires_grad=True, device='cuda')

# TODO: keep these functions in the scrits directory

# Sampling methods for the neural subdivision complex
def lerp(X, U, V):
    lp00 = X[:, 0, :].unsqueeze(1) * U.unsqueeze(-1) * V.unsqueeze(-1)
    lp01 = X[:, 1, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * V.unsqueeze(-1)
    lp10 = X[:, 3, :].unsqueeze(1) * U.unsqueeze(-1) * (1.0 - V.unsqueeze(-1))
    lp11 = X[:, 2, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * (1.0 - V.unsqueeze(-1))
    return lp00 + lp01 + lp10 + lp11

def sample(sample_rate):
    U = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    V = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    U, V = torch.meshgrid(U, V)

    corner_points = points[complexes, :]
    corner_encodings = encodings[complexes, :]

    U, V = U.reshape(-1), V.reshape(-1)
    U = U.repeat((complexes.shape[0], 1))
    V = V.repeat((complexes.shape[0], 1))

    lerped_points = lerp(corner_points, U, V).reshape(-1, 3)
    lerped_encodings = lerp(corner_encodings, U, V).reshape(-1, POINT_ENCODING_SIZE)

    return lerped_points, lerped_encodings

def indices(sample_rate):
    triangles = []
    for i in range(sample_rate - 1):
        for j in range(sample_rate - 1):
            a = i * sample_rate + j
            c = (i + 1) * sample_rate + j
            b, d = a + 1, c + 1
            triangles.append([a, c, b])
            triangles.append([b, c, d])

    return np.array(triangles, dtype=np.int32)

def sample_rate_indices(sample_rate):
    tri_indices = []
    for i in range(complexes.shape[0]):
        ind = indices(sample_rate)
        ind += i * sample_rate ** 2
        tri_indices.append(ind)

    tri_indices = np.concatenate(tri_indices, axis=0)
    tri_indices_tensor = torch.from_numpy(tri_indices).cuda()
    return tri_indices_tensor

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

def average_edge_length(V, T):
    v0 = V[T[:, 0], :]
    v1 = V[T[:, 1], :]
    v2 = V[T[:, 2], :]

    v01 = v1 - v0
    v02 = v2 - v0
    v12 = v2 - v1

    l01 = torch.norm(v01, dim=1)
    l02 = torch.norm(v02, dim=1)
    l12 = torch.norm(v12, dim=1)
    return (l01 + l02 + l12).mean()/3.0

import matplotlib.pyplot as plt

from render import NVDRenderer

renderer = NVDRenderer(scene_parameters, shading=True, boost=3)

def preview(V, N, F, batch, title='Preview'):
    assert len(views) % batch == 0
    fig, axs = plt.subplots(len(views) // batch, batch, figsize=(batch * 4, len(views) // batch * 4))
    for i in range(len(views) // batch):
        imgs = renderer.render(V, N, F, torch.arange(i * batch, (i + 1) * batch))
        for j in range(batch):
            ax = axs[i, j]
            ax.imshow((imgs[j].clip(0,1).pow(1/2.2)).cpu().numpy(), origin='lower')
            ax.axis('off')

    fig.suptitle(title)
    fig.tight_layout()

def preview_nsc(sample_rate, batch, title='Preview (NSC)'):
    LP, LE = sample(sample_rate)
    V = m(points=LP, features=LE)

    V = V.detach()
    F = sample_rate_indices(sample_rate)
    Fn = compute_face_normals(V, F)
    N = compute_vertex_normals(V, F, Fn)

    preview(V, N, F, batch=batch, title=title)

#preview(v_ref, n_ref, f_ref, batch=splitsPhi, title='Preview (Ground Truth)')
#preview_nsc(4, batch=splitsPhi, title='Preview (NSC)')

from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform

from torch.utils.tensorboard import SummaryWriter

def preview_tb(V, N, F, batch, title='Preview'):
    assert len(views) % batch == 0
    fig, axs = plt.subplots(len(views) // batch, batch, figsize=(batch * 4, len(views) // batch * 4))
    for i in range(len(views) // batch):
        imgs = renderer.render(V, N, F, torch.arange(i * batch, (i + 1) * batch))
        for j in range(batch):
            ax = axs[i, j]
            ax.imshow((imgs[j].clip(0,1).pow(1/2.2)).cpu().numpy(), origin='lower')
            ax.axis('off')

    fig.suptitle(title)
    fig.tight_layout()
    return fig

steps     = 1000   # Number of optimization steps
step_size = 1e-2   # Step size
lambda_   = 20     # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_ * L)

target_cas = casdf_cpp.geometry(v_ref.cpu(), f_ref.cpu())
target_cas = casdf_cpp.cas_grid(target_cas, 32)

writer = SummaryWriter('runs/nefertiti')

def inverse_render(sample_rate, use_LP=False):
    print(f'Inverse rendering at sample rate {sample_rate}...')

    # Get the reference images
    # ref_imgs = renderer.render(v_ref, n_ref, f_ref)

    # Compute the system matrix and parameterize
    LP, LE = sample(sample_rate)
    V = m(points=LP, features=LE).detach()
    if use_LP:
        V = LP.detach()

    # Fix indices...
    cmap = make_cmap(complexes, LP, sample_rate)
    F, remap = geom_cpp.sdc_weld(complexes.cpu(), cmap, V.shape[0], sample_rate)
    F = F.cuda()

    # Setup optimization
    M = compute_matrix(V, F, lambda_)
    U = to_differential(M, V)

    U.requires_grad = True
    opt = AdamUniform([ U ], step_size)

    Fn = compute_face_normals(V, F)
    N = compute_vertex_normals(V, F, Fn)

    #preview(v_ref, n_ref, f_ref, batch=splitsPhi, title='Reference')
    #preview(V, N.detach(), F, batch=splitsPhi, title='Initial')

    # Optimization loop
    losses =  {}

    batch_size = len(views)
    assert len(views) % batch_size == 0
    print(f'Batch size: {batch_size}')

    for it in trange(steps):
        summed = []
        indices = torch.arange(0, len(views)).cuda()

        # Permute randomly, then cut up into batches
        batches = int(len(views) / batch_size)
        indices = indices[torch.randperm(len(indices))]
        indices = indices[:batches * batch_size].reshape(batches, batch_size)

        for bi in indices:
            # Get cartesian coordinates for parameterization
            V = from_differential(M, U, 'Cholesky')

            # Recompute vertex normals
            Fn = compute_face_normals(V, F)
            N = compute_vertex_normals(V, F, Fn)

            # Render images
            # indices = torch.arange(b * splitsPhi, (b + 1) * splitsPhi).cuda()
            ref_imgs = renderer.render(v_ref, n_ref, f_ref, bi)
            opt_imgs = renderer.render(V, N, F, bi)

            # Compute losses
            img_loss = (opt_imgs - ref_imgs).abs().mean()
            loss = img_loss

            summed.append(img_loss.item())

            # Optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Write to tensorboard
        writer.add_scalar(f'render-loss-{sample_rate}', np.mean(summed), it)
        writer.flush()

    # TODO: render to tensorboard
    # preview(V.detach(), N.detach(), F, batch=splitsPhi, title='Optimized')
    fig = preview_tb(V.detach(), N.detach(), F, batch=splitsPhi, title='Optimized')
    writer.add_figure(f'optimized-{sample_rate}', fig)

    V = geom_cpp.sdc_separate(V.detach().cpu(), remap).cuda()

    for k, v in losses.items():
        losses[k] = np.array(v)

    return V

def train_nsc(V, sample_rate):
    print(f'Training Neural Subdivision Complex at sample rate {sample_rate}...')
    optimizer = torch.optim.Adam(list(m.parameters()) + [ encodings ], lr=1e-3)
    # F = sample_rate_indices(sample_rate)

    LP, LE = sample(sample_rate)
    cmap = make_cmap(complexes, LP, sample_rate)
    F, remap = geom_cpp.sdc_weld(complexes.cpu(), cmap, V.shape[0], sample_rate)
    F = F.cuda()

    aell = average_edge_length(V, F).detach()

    Fn = compute_face_normals(V, F)
    N = compute_vertex_normals(V, F, Fn)

    history = {}
    for _ in trange(1_000):
        LP, LE = sample(sample_rate)
        X = m(points=LP, features=LE)

        vertex_loss = (X - V).square().mean()

        XFn = compute_face_normals(X, F)
        Xn = compute_vertex_normals(X, F, XFn)

        normal_loss = (Xn - N).square().mean()
        loss = vertex_loss + aell * normal_loss

        # TODO: seam loss: compile once, apply everywhere
        #history.setdefault('loss', []).append(loss.item())
        #history.setdefault('vertex loss', []).append(vertex_loss.item())
        #history.setdefault('normal loss', []).append((aell * normal_loss).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar(f'loss-{sample_rate}', loss.item(), _)
        writer.add_scalar(f'vertex-loss-{sample_rate}', vertex_loss.item(), _)
        writer.add_scalar(f'normal-loss-{sample_rate}', normal_loss.item(), _)
        writer.flush()

    preview(v_ref, n_ref, f_ref, batch=splitsPhi, title='Reference')
    preview_nsc(sample_rate, batch=splitsPhi, title='Optimized')

    return X.detach()

import polyscope as ps

ps.init()

def color_code_complexes(C, sample_rate=16):
    color_wheel = [
            np.array([0.700, 0.300, 0.300]),
            np.array([0.700, 0.450, 0.300]),
            np.array([0.700, 0.600, 0.300]),
            np.array([0.650, 0.700, 0.300]),
            np.array([0.500, 0.700, 0.300]),
            np.array([0.350, 0.700, 0.300]),
            np.array([0.300, 0.700, 0.400]),
            np.array([0.300, 0.700, 0.550]),
            np.array([0.300, 0.700, 0.700]),
            np.array([0.300, 0.550, 0.700]),
            np.array([0.300, 0.400, 0.700]),
            np.array([0.350, 0.300, 0.700]),
            np.array([0.500, 0.300, 0.700]),
            np.array([0.650, 0.300, 0.700]),
            np.array([0.700, 0.300, 0.600]),
            np.array([0.700, 0.300, 0.450])
    ]

    complex_face_colors = []
    for i in range(C.shape[0]):
        color = color_wheel[i % len(color_wheel)]
        complex_face_colors.append(np.tile(color, (2 * (sample_rate - 1) ** 2, 1)))

    return np.concatenate(complex_face_colors)

for sample_rate in [ 4, 8, 16 ]:
    V = inverse_render(sample_rate, use_LP=(sample_rate == 4))
    X = train_nsc(V, sample_rate)

    I = sample_rate_indices(sample_rate)
    colors = color_code_complexes(complexes, sample_rate=sample_rate)
    ps.register_surface_mesh(f'Vir-{sample_rate}', V.detach().cpu().numpy(), I.cpu().numpy()).add_color_quantity('colors', colors, defined_on='faces')
    ps.register_surface_mesh(f'learned-{sample_rate}', X.detach().cpu().numpy(), I.cpu().numpy()).add_color_quantity('colors', colors, defined_on='faces')

ps.show()
