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

v_ref  = torch.from_numpy(mesh.points).float().cuda()
f_ref  = torch.from_numpy(mesh.cells_dict['triangle']).int().cuda()
fn_ref = compute_face_normals(v_ref, f_ref)
n_ref  = compute_vertex_normals(v_ref, f_ref, fn_ref)

print('vertices:', v_ref.shape, 'faces:', f_ref.shape)

# Generate camera views
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

def sample_cameras(batch):
    # TODO: pass an offset to the camera
    triangles = torch.randint(0, f_ref.shape[0], (batch,), device='cuda')
    barys = torch.rand((batch, 2), device='cuda')
    barys = torch.where(barys.sum(dim=1, keepdim=True) > 1.0, 1.0 - barys, barys)
    barys = torch.cat((barys, 1.0 - barys.sum(dim=1, keepdim=True)), dim=1)

    v0 = v_ref[f_ref[triangles, 0]]
    v1 = v_ref[f_ref[triangles, 1]]
    v2 = v_ref[f_ref[triangles, 2]]

    n0 = n_ref[f_ref[triangles, 0]]
    n1 = n_ref[f_ref[triangles, 1]]
    n2 = n_ref[f_ref[triangles, 2]]

    view_points = barys[:, 0].unsqueeze(1) * v0 + barys[:, 1].unsqueeze(1) * v1 + barys[:, 2].unsqueeze(1) * v2
    view_normals = barys[:, 0].unsqueeze(1) * n0 + barys[:, 1].unsqueeze(1) * n1 + barys[:, 2].unsqueeze(1) * n2
    view_normals = view_normals / torch.norm(view_normals, dim=1, keepdim=True)

    eye_offsets = view_normals * 1.0
    eyes = view_points + eye_offsets

    canonical_up = torch.tensor([0.0, 1.0, 0.0], device='cuda')
    canonical_up = torch.stack(batch * [canonical_up], dim=0)

    rights = torch.cross(view_normals, canonical_up)
    ups = torch.cross(rights, view_normals)

    views = [ lookat(eye, view_point, up) for eye, view_point, up in zip(eyes, view_points, ups) ]
    return torch.stack(views, dim=0), eyes, view_normals

all_views, eyes, forwards = sample_cameras(100)
import polyscope as ps
ps.init()
ps.register_surface_mesh('mesh', v_ref.cpu().numpy(), f_ref.cpu().numpy())
ps.register_point_cloud('views', eyes.cpu().numpy()) \
        .add_vector_quantity('forwards', -forwards.cpu().numpy(), enabled=True)
ps.show()

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

# TODO: tone mapping

def inverse_render(V, F):
    steps     = 1_000  # Number of optimization steps
    step_size = 1e-2   # Step size
    lambda_   = 10     # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_ * L)

    # Get the reference images
    # ref_imgs = renderer.render(v_ref, n_ref, f_ref)
    # ref_imgs = ref_imgs
    batch = 10

    # Optimization setup
    M = compute_matrix(V, F, lambda_)
    U = to_differential(M, V)

    U.requires_grad = True
    opt = AdamUniform([ U ], step_size)

    # Optimization loop
    for it in trange(steps):
        # Batch the views into disjoint sets
        assert len(all_views) % batch == 0
        views = torch.split(all_views, batch, dim=0)
        # print('Batched into', views)
        # views = [ all_views[i:i + batch] for i in range(0, len(all_views), batch) ]

        for i, view_mats in enumerate(views):
            V = from_differential(M, U, 'Cholesky')

            Fn = compute_face_normals(V, F)
            N = compute_vertex_normals(V, F, Fn)

            opt_imgs = renderer.render(V, N, F, view_mats)
            ref_imgs = renderer.render(v_ref, n_ref, f_ref, view_mats)

            # Show render
            # if i == 0:
            #     fig, ax = plt.subplots(1, 2)
            #     ax[0].imshow(ref_imgs[0].pow(1/2.2).cpu().numpy())
            #     ax[1].imshow(opt_imgs[0].pow(1/2.2).detach().cpu().numpy())
            #     plt.show()

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
ps.show()

# Inverse rendering from here...

# Get the reference images
def inverse_render_nsc(sample_rate, losses, lr):
    # ref_imgs = renderer.render(v_ref, n_ref, f_ref)
    opt      = torch.optim.Adam(list(m.parameters()) + [ features ], lr)

    base, _  = sample(complexes, points, features, sample_rate)
    cmap     = make_cmap(complexes, points, base, sample_rate)
    remap    = optext.generate_remapper(complexes.cpu(), cmap, base.shape[0], sample_rate)
    batch    = 10

    for _ in trange(1_000):
        # Batch the views into disjoint sets
        assert len(all_views) % batch == 0
        views = torch.split(all_views, batch, dim=0)

        for view_mats in views:
            lerped_points, lerped_features = sample(complexes, points, features, sample_rate)
            V = m(points=lerped_points, features=lerped_features)

            indices = optext.triangulate_shorted(V, complexes.shape[0], sample_rate)
            F = remap.remap_device(indices)
            Fn = compute_face_normals(V, F)
            N = compute_vertex_normals(V, F, Fn)

            opt_imgs = renderer.render(V, N, F, view_mats)
            ref_imgs = renderer.render(v_ref, n_ref, f_ref, view_mats)

            # Compute losses
            # TODO: tone mapping from NVIDIA paper
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

V, F, losses = inverse_render_nsc(4, {}, 1e-3)
V, F, losses = inverse_render_nsc(8, losses, 1e-3)
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
