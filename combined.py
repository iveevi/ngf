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

assert len(sys.argv) >= 5, 'Usage: python combined.py <directory> <model> <kernel> <clusters>'

# Load all necessary extensions
if not os.path.exists('build'):
    os.makedirs('build')

optext = load(name='optext',
        sources=[ 'optext.cu' ],
        extra_include_paths=[ 'glm' ],
        build_directory='build',
        extra_cflags=[ '-O3' ],
        extra_cuda_cflags=[ '-O3' ])

print('Loaded optimization extension')

directory = sys.argv[1]
mesh = meshio.read(os.path.join(directory, 'target.obj'))

environment = imageio.imread('images/environment.hdr', format='HDR-FI')
environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
alpha       = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
environment = torch.cat((environment, alpha), dim=-1)

scene_parameters = {}
scene_parameters['res_x']        = 1024
scene_parameters['res_y']        = 640
scene_parameters['fov']          = 45.0
scene_parameters['near_clip']    = 0.1
scene_parameters['far_clip']     = 1000.0
scene_parameters['envmap']       = environment
scene_parameters['envmap_scale'] = 1.0

v_ref  = torch.from_numpy(mesh.points).float().cuda()
f_ref  = torch.from_numpy(mesh.cells_dict['triangle']).int().cuda()
fn_ref = compute_face_normals(v_ref, f_ref)
n_ref  = compute_vertex_normals(v_ref, f_ref, fn_ref)

print('vertices:', v_ref.shape, 'faces:', f_ref.shape)

# Simplify the target mesh for simplification
simplify_binary = './build/simplify'
reduction       = 10_000/f_ref.shape[0]
result          = os.path.join(directory, 'simplified.obj')

print('Simplifying mesh with qslim...')
print('Reduction:', reduction)
print('Result:', result)
os.system('{} {} {} {:4f}'.format(simplify_binary, os.path.join(directory, 'target.obj'), result, reduction))

simplified_mesh = meshio.read(result)
v_simplified    = torch.from_numpy(simplified_mesh.points).float().cuda()
f_simplified    = torch.from_numpy(simplified_mesh.cells_dict['triangle']).int().cuda()
fn_simplified   = compute_face_normals(v_simplified, f_simplified)
n_simplified    = compute_vertex_normals(v_simplified, f_simplified, fn_simplified)

cameras         = int(sys.argv[4])
seeds           = list(torch.randint(0, f_simplified.shape[0], (cameras,)).numpy())
target_geometry = optext.geometry(v_simplified.cpu(), n_simplified.cpu(), f_simplified.cpu())
target_geometry = target_geometry.deduplicate()
clusters        = optext.cluster_geometry(target_geometry, seeds, 10)

# Compute scene extents for camera placement
min = v_ref.min(dim=0)[0]
max = v_ref.max(dim=0)[0]
extent = (max - min).square().sum().sqrt().item()
print('Extent:', extent)

# Compute the centroid and normal for each cluster
cluster_centroids = []
cluster_normals = []

for cluster in clusters:
    faces = f_simplified[cluster]

    v0 = v_simplified[faces[:, 0]]
    v1 = v_simplified[faces[:, 1]]
    v2 = v_simplified[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    centroids = centroids.mean(dim=0)

    normals = torch.cross(v1 - v0, v2 - v0)
    normals = normals.mean(dim=0)
    normals = normals / torch.norm(normals)

    cluster_centroids.append(centroids)
    cluster_normals.append(normals)

cluster_centroids = torch.stack(cluster_centroids, dim=0)
cluster_normals = torch.stack(cluster_normals, dim=0)

# Generate camera views
canonical_up = torch.tensor([0.0, 1.0, 0.0], device='cuda')
cluster_eyes = cluster_centroids + cluster_normals * 0.1 * extent
cluster_ups = torch.stack(len(clusters) * [ canonical_up ], dim=0)
cluster_rights = torch.cross(cluster_normals, cluster_ups)
cluster_ups = torch.cross(cluster_rights, cluster_normals)

all_views = [ lookat(eye, view_point, up) for eye, view_point, up in zip(cluster_eyes, cluster_centroids, cluster_ups) ]
all_views = torch.stack(all_views, dim=0)

import polyscope as ps
ps.init()
ps.register_surface_mesh('mesh', v_ref.cpu().numpy(), f_ref.cpu().numpy())
# ps.register_point_cloud('views', eyes.cpu().numpy()) \
#         .add_vector_quantity('forwards', -forwards.cpu().numpy(), enabled=True)
ps.register_point_cloud('clustered views', cluster_eyes.cpu().numpy()) \
        .add_vector_quantity('forwards', -cluster_normals.cpu().numpy(), enabled=True)
for i, c in enumerate(clusters):
    ps.register_surface_mesh('cluster_{}'.format(i), v_simplified.cpu().numpy(), f_simplified.cpu().numpy()[c])
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
    batch     = 10

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
from configurations import *

m = models[sys.argv[2]]().cuda()

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
ps.register_point_cloud('clustered views', cluster_eyes.cpu().numpy()) \
        .add_vector_quantity('forwards', -cluster_normals.cpu().numpy(), enabled=True)
# ps.show()

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

        batch_losses = []
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
            loss = render_loss # + 1e3 * area_loss

            # Optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()

            # losses.setdefault('render', []).append(render_loss.item())
            # losses.setdefault('area', []).append(area_loss.item())
            batch_losses.append(loss.item())

        losses.setdefault('loss', []).append(sum(batch_losses) / len(batch_losses))

    return V.detach(), F, losses

V, F, losses = inverse_render_nsc(4, {}, 1e-3)
V, F, losses = inverse_render_nsc(8, losses, 1e-3)
V, F, losses = inverse_render_nsc(16, losses, 1e-3)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.plot(losses['loss'], label='loss')
# plt.plot(losses['render'], label='render')
# plt.plot(losses['area'], label='area')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')

plt.title('Losses')
plt.tight_layout()
plt.legend()

# plt.savefig('losses.png')

ps.register_surface_mesh('model-phase2', V.cpu().numpy(), F.cpu().numpy())
# ps.show()

# Save data
result = os.path.basename(sys.argv[1])
result = os.path.splitext(result)[0]
result = os.path.join('results', result, sys.argv[2] + '-' + sys.argv[3])
os.makedirs(result, exist_ok=True)

model = {
    'model': m,
    'features': features,
    'complexes': complexes,
    'points': points,
}

torch.save(model, os.path.join(result, 'model.pt'))
plt.savefig(os.path.join(result, 'loss.png'))
