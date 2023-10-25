import os
import seaborn as sns
import sys
import torch
import matplotlib.pyplot as plt
import meshio
import imageio

from scripts.geometry import compute_vertex_normals, compute_face_normals
from scripts.load_xml import load_scene

from torch.utils.cpp_extension import load
from tqdm import trange

from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform

from scripts.render import NVDRenderer

from mlp import *
from util import *

sns.set()

print('Loading optext library...')

assert len(sys.argv) > 1
if not os.path.exists('build'):
    os.makedirs('build')

# optext = load(name='geom_cpp',
#         sources=[ 'ext/optext.cpp' ],
#         extra_include_paths=[ 'glm' ],
#         build_directory='build')

optext = load(name='optext', sources=[ 'ext/casdf.cu' ], extra_include_paths=[ 'glm' ], build_directory='build')

directory = sys.argv[1]
mesh = meshio.read(os.path.join(directory, 'target.obj'))

environment = imageio.imread('images/environment.hdr', format='HDR-FI')
environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
alpha = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
environment = torch.cat((environment, alpha), dim=-1)

scene_parameters = {}
scene_parameters['res_x'] = 1024
scene_parameters['res_y'] = 640
scene_parameters['fov'] = 45.0
scene_parameters['near_clip'] = 0.1
scene_parameters['far_clip'] = 1000.0
scene_parameters['envmap'] = environment
scene_parameters['envmap_scale'] = 1.0

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

if len(sys.argv) > 2:
    sp = load_scene(sys.argv[2])
    scene_parameters['view_mats'] = sp['view_mats']
else:
    views = []

    # TODO: cluster the faces and then use normal offset...

    # Try smaller radius...
    # radius = 5.0
    splitsTheta = 5
    splitsPhi = 5

    center = v_ref.mean(axis=0)
    radius = 2 * torch.norm(v_ref - center, dim=1).max()

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

mesh = meshio.read(os.path.join(directory, 'source.obj'))

v = mesh.points
f = mesh.cells_dict['quad']

print('Quadrangulated shape; vertices:', v.shape, 'faces:', f.shape)

# Configure neural subdivision complex parameters
points = torch.from_numpy(v).float().cuda()
complexes = torch.from_numpy(f).int().cuda()
features = torch.zeros((points.shape[0], POINT_ENCODING_SIZE), requires_grad=True, device='cuda')

# Setup the rendering backend
renderer = NVDRenderer(scene_parameters, shading=True, boost=3)

# Initial (4x4) optimization
import polyscope as ps

base, _ = sample(complexes, points, features, 4)

base_indices = shorted_indices(base.cpu().numpy(), complexes.cpu().numpy(), 4)
tch_base_indices = torch.from_numpy(base_indices).int()

cmap = make_cmap(complexes, points, base, 4)
remap = optext.generate_remapper(complexes.cpu(), cmap, base.shape[0], 4)
F = remap.remap(tch_base_indices).cuda()
print('F:', F.shape)

Fi = optext.triangulate_shorted(base, complexes.shape[0], 4)

ps.init()
ps.register_surface_mesh('reference', v_ref.cpu().numpy(), f_ref.cpu().numpy())
ps.register_surface_mesh('base-original', base.cpu().numpy(), Fi.cpu().numpy())

def inverse_render(V, F):
    steps     = 1000   # Number of optimization steps
    step_size = 1e-2   # Step size
    lambda_   = 10     # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_ * L)

    # Get the reference images
    ref_imgs = renderer.render(v_ref, n_ref, f_ref)
    ref_imgs = ref_imgs

    # Optimization setup
    M = compute_matrix(V, F, lambda_)
    U = to_differential(M, V)

    U.requires_grad = True
    opt = AdamUniform([ U ], step_size)

    # Optimization loop
    for it in trange(steps):
        V = from_differential(M, U, 'Cholesky')

        Fn = compute_face_normals(V, F)
        N = compute_vertex_normals(V, F, Fn)

        opt_imgs = renderer.render(V, N, F)
        opt_imgs = opt_imgs

        # Compute losses
        # TODO: tone mapping from NVIDIA paper
        render_loss = (opt_imgs - ref_imgs).abs().mean()
        area_loss = triangle_areas(V, F).var()
        loss = render_loss + 1e3 * area_loss

        # Optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()

    return V

base = inverse_render(base, F)
base = remap.scatter(base.cpu()).cuda()

# Train model to the base first
m = MLP_Positional_Encoding().cuda()
opt = torch.optim.Adam(list(m.parameters()) + [ features ], 1e-2)
for _ in trange(1000):
    lerped_points, lerped_features = sample(complexes, points, features, 4)
    V = m(points=lerped_points, features=lerped_features)
    loss = (V - base).abs().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

base_indices = shorted_indices(base.cpu().numpy(), complexes.cpu().numpy(), 4)
ps.register_surface_mesh('base', base.cpu().numpy(), base_indices)

V = V.detach()
indices = shorted_indices(V.cpu().numpy(), complexes.cpu().numpy(), 4)
ps.register_surface_mesh('model-phase1', V.cpu().numpy(), indices)

# Inverse rendering from here...

# Get the reference images
def inverse_render_nsc(sample_rate, losses, lr):
    ref_imgs = renderer.render(v_ref, n_ref, f_ref)
    opt      = torch.optim.Adam(list(m.parameters()) + [ features ], lr)

    base, _  = sample(complexes, points, features, sample_rate)
    cmap     = make_cmap(complexes, points, base, sample_rate)
    remap    = optext.generate_remapper(complexes.cpu(), cmap, base.shape[0], sample_rate)

    for _ in trange(5_000):
        lerped_points, lerped_features = sample(complexes, points, features, sample_rate)
        V = m(points=lerped_points, features=lerped_features)

        indices = optext.triangulate_shorted(V, complexes.shape[0], sample_rate)
        F = remap.remap_device(indices)
        Fn = compute_face_normals(V, F)
        N = compute_vertex_normals(V, F, Fn)

        opt_imgs = renderer.render(V, N, F)

        # Compute losses
        render_loss = (opt_imgs - ref_imgs).abs().mean()
        area_loss = triangle_areas(V, F).var()
        loss = render_loss + 1e3 * area_loss

        # Optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.setdefault('render', []).append(render_loss.item())
        losses.setdefault('area', []).append(area_loss.item())

    return V.detach(), F, losses

V, F, losses = inverse_render_nsc(4, {}, 1e-2)
V, F, losses = inverse_render_nsc(8, losses, 5e-3)
V, F, losses = inverse_render_nsc(16, losses, 1e-3)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.plot(losses['render'], label='render')
# plt.plot(losses['area'], label='area')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')

plt.title('Losses')
plt.tight_layout()
plt.legend()

plt.savefig('losses.png')

ps.register_surface_mesh('model-phase2', V.cpu().numpy(), F.cpu().numpy())

ps.show()
