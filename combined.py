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

assert len(sys.argv) == 8, 'Usage: python combined.py <target> <source> <method> <clusters> <batch> <resolution> <result>'

# Parse all the arguments
target     = sys.argv[1]
source     = sys.argv[2]
method     = sys.argv[3].split('/')
batch      = eval(sys.argv[5])
resolution = int(sys.argv[6])
result     = sys.argv[7]

if type(batch) == int:
    batch_size = batch
    batch = lambda r: batch_size

assert len(method) == 2, 'Method must be of the form <activation + encoding>/<interpolation>'

print('Training neural subdivision complexes with the following configuration')
print('  > Target:    ', target)
print('  > Source:    ', source)
print('  > Method:    ', method)
print('  > Batch:     ', batch)
print('  > Resolution:', resolution)
print('  > Result:    ', result)

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

mesh = meshio.read(target)

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
base            = os.path.dirname(os.path.realpath(target))
simplify_binary = './build/simplify'
reduction       = 10_000/f_ref.shape[0]
simplified_path = os.path.join(base, 'simplified.obj')

print('Simplifying mesh with qslim...')
print('Reduction:', reduction)
print('Result:', simplified_path)
os.system('{} {} {} {:4f}'.format(simplify_binary, target, simplified_path, reduction))

simplified_mesh = meshio.read(simplified_path)
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
center = (min + max) / 2.0
extent = (max - min).square().sum().sqrt() / 2.0
print('Extent:', extent)

normalize = lambda x: (x - center) / extent

v_ref = normalize(v_ref)
print('new min:', v_ref.min(dim=0)[0])
print('new max:', v_ref.max(dim=0)[0])

v_simplified = normalize(v_simplified)

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
cluster_eyes = cluster_centroids + cluster_normals * 1.0
cluster_ups = torch.stack(len(clusters) * [ canonical_up ], dim=0)
cluster_rights = torch.cross(cluster_normals, cluster_ups)
cluster_ups = torch.cross(cluster_rights, cluster_normals)

all_views = [ lookat(eye, view_point, up) for eye, view_point, up in zip(cluster_eyes, cluster_centroids, cluster_ups) ]
all_views = torch.stack(all_views, dim=0)

# Configure the model
mesh = meshio.read(source)

v = mesh.points
f = mesh.cells_dict['quad']

print('Complexes; vertices:', v.shape, 'faces:', f.shape)

# Configure neural subdivision complex parameters
points = torch.from_numpy(v).float().cuda()
complexes = torch.from_numpy(f).int().cuda()
features = torch.zeros((points.shape[0], POINT_ENCODING_SIZE), requires_grad=True, device='cuda')

points = normalize(points)

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

# Fi = optext.triangulate_shorted(base, complexes.shape[0], 4)

ps.init()

indices = quadify(complexes.cpu().numpy(), 4)

ps.register_surface_mesh('reference', v_ref.cpu().numpy(), f_ref.cpu().numpy())
ps.register_surface_mesh('base-original', base.cpu().numpy(), indices)
ps.register_point_cloud('clustered views', cluster_eyes.cpu().numpy()) \
        .add_vector_quantity('forwards', -cluster_normals.cpu().numpy(), enabled=True)

# ps.show()

# TODO: tone mapping

compute_Fn = torch.compile(compute_face_normals, mode='reduce-overhead')
compute_N = torch.compile(compute_vertex_normals, mode='reduce-overhead')

def inverse_render(V, F, sample_rate=4):
    steps     = 1_000  # Number of optimization steps
    step_size = 3e-2   # Step size
    lambda_   = 10     # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_ * L)

    # Optimization setup
    M = compute_matrix(V, F, lambda_)
    U = to_differential(M, V)

    U.requires_grad = True
    opt = AdamUniform([ U ], step_size)

    indices = quadify(complexes.cpu().numpy(), sample_rate)
    vgraph = optext.vertex_graph(F.cpu())
    cgraph = optext.conformal_graph(torch.from_numpy(indices).int())
    print('vgraph:', vgraph, 'cgraph:', cgraph)

    a, opp_a = cgraph[:, 0], cgraph[:, 1]
    b, opp_b = cgraph[:, 2], cgraph[:, 3]
    print('a:', a.shape, 'opp_a:', opp_a.shape)
    print('b:', b.shape, 'opp_b:', opp_b.shape)

    # Optimization loop
    batch_size = batch(sample_rate)
    for it in trange(steps):
        # Batch the views into disjoint sets
        assert len(all_views) % batch_size == 0
        views = torch.split(all_views, batch_size, dim=0)
        for i, view_mats in enumerate(views):
            V = from_differential(M, U, 'Cholesky')

            Fn = compute_Fn(V, F)
            N = compute_N(V, F, Fn)

            opt_imgs = renderer.render(V, N, F, view_mats)
            ref_imgs = renderer.render(v_ref, n_ref, f_ref, view_mats)

            V_smoothed = vgraph.smooth_device(V, 1.0)
            V_smoothed = vgraph.smooth_device(V_smoothed, 1.0)

            # Compute losses
            # TODO: tone mapping from NVIDIA paper
            render_loss = (opt_imgs - ref_imgs).abs().mean()
            laplacian_loss = (V - V_smoothed).abs().mean()
            loss = render_loss + laplacian_loss

            # Optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()

    return V

base = inverse_render(base, F)
base = remap.scatter(base.cpu()).cuda()

# Train model to the base first
from configurations import *

m = models[method[0]]().cuda()
c = clerp(lerps[method[1]])
s = sampler(kernel=c)

opt = torch.optim.Adam(list(m.parameters()) + [ features ], 1e-2)
for _ in trange(1000):
    lerped_points, lerped_features = s(complexes, points, features, 4)
    V = m(points=lerped_points, features=lerped_features)
    loss = (V - base).abs().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

# base_indices = shorted_indices(base.cpu().numpy(), complexes.cpu().numpy(), 4)

V = V.detach().cpu().numpy()
# indices = shorted_indices(V.cpu().numpy(), complexes.cpu().numpy(), 4)
indices = quadify(complexes.cpu().numpy(), 4)

ps.register_surface_mesh('model-phase1', V, indices)
ps.register_surface_mesh('base', base.cpu().numpy(), indices)
ps.register_point_cloud('clustered views', cluster_eyes.cpu().numpy()) \
        .add_vector_quantity('forwards', -cluster_normals.cpu().numpy(), enabled=True)

# ps.show()

# Get the reference images
def inverse_render_nsc(sample_rate, losses, lr, aux_strength=1.0):
    opt      = torch.optim.Adam(list(m.parameters()) + [ features ], lr)
    base, _  = sample(complexes, points, features, sample_rate)
    cmap     = make_cmap(complexes, points, base, sample_rate)
    remap    = optext.generate_remapper(complexes.cpu(), cmap, base.shape[0], sample_rate)
    vgraph   = None

    batch_size = batch(sample_rate)
    for i in trange(1_000):
        # Batch the views into disjoint sets
        assert len(all_views) % batch_size == 0
        views = torch.split(all_views, batch_size, dim=0)

        if i % 50 == 0:
            # print('Rebuilding vgraph...')
            lerped_points, lerped_features = s(complexes, points, features, sample_rate)
            V = m(points=lerped_points, features=lerped_features)
            indices = optext.triangulate_shorted(V, complexes.shape[0], sample_rate)
            F = remap.remap_device(indices)
            vgraph = optext.vertex_graph(F.cpu())

        for view_mats in views:
            lerped_points, lerped_features = s(complexes, points, features, sample_rate)
            V = m(points=lerped_points, features=lerped_features)

            indices = optext.triangulate_shorted(V, complexes.shape[0], sample_rate)
            F = remap.remap_device(indices)

            Fn = compute_face_normals(V, F)
            N = compute_vertex_normals(V, F, Fn)

            opt_imgs = renderer.render(V, N, F, view_mats)
            ref_imgs = renderer.render(v_ref, n_ref, f_ref, view_mats)

            V_smoothed = vgraph.smooth_device(V, 1.0)
            V_smoothed = vgraph.smooth_device(V_smoothed, 1.0)

            # Compute losses
            # TODO: tone mapping from NVIDIA paper
            render_loss = (opt_imgs - ref_imgs).abs().mean()
            laplacian_loss = (V - V_smoothed).abs().mean()
            area_loss = triangle_areas(V, F).mean()
            loss = render_loss + aux_strength * (area_loss + laplacian_loss)

            # Optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()

    return V.detach(), F, losses

r = 4
aux_strength = 1.0
while r <= resolution:
    V, F, losses = inverse_render_nsc(r, {}, 1e-3, aux_strength)
    aux_strength *= 0.5
    r *= 2

    # indices = quadify(complexes.cpu().numpy(), r)
    # ps.register_surface_mesh(f'model-phase2-{r}', V.cpu().numpy(), indices)
    # ps.show()

# Save data
print('Saving model to', result)
directory = os.path.dirname(result)
os.makedirs(directory, exist_ok=True)

model = {
    'model': m,
    'features': features,
    'complexes': complexes,
    'points': points,
    'kernel': method[1],
}

torch.save(model, result)
