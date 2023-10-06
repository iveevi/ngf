import os
import sys
import seaborn as sns
import torch

from torch.utils.cpp_extension import load

sns.set()

filename = os.path.join('scenes', 'dragon', 'dragon.xml')
print(f'Loading scene from {filename}')

if not os.path.exists('scenes'):
    print('Loading scenes...')
    os.system('wget https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Nicolet2021Large.zip')
    os.system('unzip Nicolet2021Large.zip')
    os.system('rm Nicolet2021Large.zip')

print('Loading geometry library...')

if not os.path.exists('build'):
    os.makedirs('build')

geom_cpp = load(name="geom_cpp",
        sources=[ "ext/geometry.cpp" ],
        extra_include_paths=[ "glm" ],
        build_directory="build",
)

from scripts.load_xml import load_scene

scene_parameters = load_scene(filename)

v_ref = scene_parameters['mesh-target']['vertices']
n_ref = scene_parameters['mesh-target']['normals']
f_ref = scene_parameters['mesh-target']['faces']

print('vertices:', v_ref.shape, 'faces:', f_ref.shape)

import meshio
import pymeshlab

mesh = meshio.Mesh(v_ref.cpu(), [('triangle', f_ref.cpu())], { 'n': n_ref.cpu() })
mesh.write('ref.obj')

ms = pymeshlab.MeshSet()
ms.load_new_mesh('ref.obj')

face_count = f_ref.shape[0]
target_count = 750

print(f'Attempting to reduce from {face_count} faces to {target_count}')

ms.meshing_decimation_quadric_edge_collapse(
    targetfacenum=target_count,
    qualitythr=0.7,
    preservenormal=True,
    preservetopology=True,
    preserveboundary=True,
    optimalplacement=False,
)

ms.meshing_repair_non_manifold_edges()
ms.meshing_tri_to_quad_by_4_8_subdivision()
ms.save_current_mesh('quadrangulated.obj')

print('Resulting face count   ', str(ms.current_mesh().face_number()))
print('Resulting vertex count ', str(ms.current_mesh().vertex_number()))

mesh = meshio.read('quadrangulated.obj')

v = mesh.points
f = mesh.cells_dict['quad']

v = torch.from_numpy(v).float().cuda()

print('Quadrangulated shape; vertices:', v.shape, 'faces:', f.shape)

# Configure neural subdivision complex parameters
from models import *

nsc = NSubComplex().cuda()

points = v
complexes = torch.from_numpy(f).int().cuda()
encodings = torch.zeros((points.shape[0], POINT_ENCODING_SIZE), requires_grad=True, device='cuda')

# Computing initial normal vectors
from scripts.geometry import compute_vertex_normals, compute_face_normals

complex_normals = compute_face_normals(v, complexes)
n = compute_vertex_normals(v, complexes, complex_normals)

# Convert to spherical coordinates
phi = 0.5 * (torch.atan2(n[:, 1], n[:, 0])/np.pi) + 0.5
theta = torch.acos(n[:, 2])/np.pi
normals = torch.stack([phi, theta], dim=1)

points.shape, complexes.shape, encodings.shape, normals.shape, nsc.parameters()

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

from scripts.render import NVDRenderer

renderer = NVDRenderer(scene_parameters, shading=True, boost=3)

def preview(V, N, F, batch=5, title='Preview'):
    imgs = renderer.render(V, N, F)
    fig, axs = plt.subplots(1, batch, figsize=(40, 20))
    for i, ax in enumerate(axs):
        ax.set_title(title + f': {i}')
        ax.imshow((imgs[i,...,:-1].clip(0,1).pow(1/2.2)).cpu().numpy(), origin='lower')
        ax.axis('off')
    plt.axis('off')
    return imgs

def preview_nsc(sample_rate, title='Preview (NSC)'):
    LP, LE = sample(sample_rate)
    V = nsc(LP, LE)

    V = V.detach()
    F = sample_rate_indices(sample_rate)
    Fn = compute_face_normals(V, F)
    N = compute_vertex_normals(V, F, Fn)

    preview(V, N, F, title=title)

# Light optimization phase
from tqdm import trange

from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.structures import Meshes

preview_nsc(4, title='Initialized')

mref = Meshes(verts=[ v_ref ], faces=[ f_ref  ])
optimizer = torch.optim.Adam(list(nsc.parameters()) + [ encodings ], lr=1e-4)

losses = {}
for sample_rate in [ 2, 4 ]:
    # TODO: function
    # F = sample_rate_indices(sample_rate)
    
    LP, LE = sample(sample_rate)
    cmap = make_cmap(complexes, LP, sample_rate)
    F, remap = geom_cpp.sdc_weld(complexes.cpu(), cmap, LP.shape[0], sample_rate)
    F = F.cuda()

    for i in trange(5000):
        LP, LE = sample(sample_rate)
        V = nsc(LP, LE)

        mopt = Meshes(verts=[ V ], faces=[ F ])

        vertex_loss, normal_loss = chamfer_distance(
            x=mref.verts_padded(),
            y=mopt.verts_padded(),
            x_normals=mref.verts_normals_padded(),
            y_normals=mopt.verts_normals_padded(),
            abs_cosine=False)
        
        laplacian_loss = mesh_laplacian_smoothing(mopt, method="uniform")
        
        losses.setdefault('vertex', []).append(vertex_loss.item())
        losses.setdefault('normal', []).append(normal_loss.item())
        losses.setdefault('laplacian', []).append(laplacian_loss.item())

        loss = vertex_loss + 1e-3 * laplacian_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

preview(v_ref, n_ref, f_ref, title='Reference')
preview_nsc(4, title='Optimized')

fig = plt.figure(figsize=(20, 10))
plt.plot(losses['vertex'], label='vertex')
plt.plot(losses['normal'], label='normal')
plt.plot(losses['laplacian'], label='laplacian')
plt.yscale('log')
plt.legend()

from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix, laplacian_uniform
from largesteps.optimize import AdamUniform

from scripts.geometry import remove_duplicates

steps = 5000       # Number of optimization steps
step_size = 1e-2   # Step size
lambda_ = 200       # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_ * L)

def inverse_render(sample_rate):
    print(f'Inverse rendering at sample rate {sample_rate}...')

    # Get the reference images
    ref_imgs = renderer.render(v_ref, n_ref, f_ref)
    
    # Compute the system matrix and parameterize
    LP, LE = sample(sample_rate)
    V = nsc(LP, LE).detach()

    cmap = make_cmap(complexes, LP, sample_rate)
    F, remap = geom_cpp.sdc_weld(complexes.cpu(), cmap, V.shape[0], sample_rate)
    F = F.cuda()

    M = compute_matrix(V, F, lambda_)
    U = to_differential(M, V)

    U.requires_grad = True
    opt = AdamUniform([ U ], step_size)

    # Optimization loop
    losses =  {}
    for it in trange(steps):

        # Get cartesian coordinates for parameterization
        V = from_differential(M, U, 'Cholesky')

        # Recompute vertex normals
        Fn = compute_face_normals(V, F)
        N = compute_vertex_normals(V, F, Fn)

        # Render images
        opt_imgs = renderer.render(V, N, F)

        # Compute losses
        img_loss = (opt_imgs - ref_imgs).abs().mean()
        loss = img_loss

        losses.setdefault('img', []).append(img_loss.item())

        # Backpropagate
        opt.zero_grad()
        loss.backward()
        
        # Update parameters
        opt.step()

    V = geom_cpp.sdc_separate(V.detach().cpu(), remap).cuda()

    print('Restored faces shape:', F.shape)

    preview(v_ref, n_ref, f_ref, title='Reference')
    preview(V, N.detach(), F, title='Optimized')

    fig = plt.figure(figsize=(20, 10))
    plt.plot(losses['img'], label='img')
    plt.yscale('log')
    plt.legend()

    return V, sample_rate

from pytorch3d.loss import mesh_normal_consistency

def train_nsc(V, sample_rate):
    print(f'Training Neural Subdivision Complex at sample rate {sample_rate}...')
    optimizer = torch.optim.Adam(list(nsc.parameters()) + [ encodings ], lr=1e-3)
    F = sample_rate_indices(sample_rate)
    aell = average_edge_length(V, F).detach()

    history = {}
    for it in trange(10000):
        LP, LE = sample(sample_rate)
        X = nsc(LP, LE)

        mopt = Meshes(verts=[ X ], faces=[ F ])

        vertex_loss = (X - V).square().mean()

        XFn = compute_face_normals(X, F)
        Fn = compute_face_normals(V, F)
        normal_loss = (XFn - Fn).square().mean()
        consistency_loss = mesh_normal_consistency(mopt)
        
        loss = vertex_loss + aell * normal_loss + aell * consistency_loss

        # TODO: seam loss: compile once, apply everywhere

        history.setdefault('loss', []).append(loss.item())
        history.setdefault('vertex loss', []).append(vertex_loss.item())
        history.setdefault('normal loss', []).append((aell * normal_loss).item())
        # history.setdefault('consistency loss', []).append((1e-6 * consistency_loss).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preview(v_ref, n_ref, f_ref, title='Reference')
    preview_nsc(sample_rate, title='Optimized')

    fig = plt.figure(figsize=(20, 10))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['vertex loss'], label='vertex loss')
    plt.plot(history['normal loss'], label='normal loss')
    # plt.plot(history['consistency loss'], label='consistency loss')
    plt.yscale('log')
    plt.legend()

    return X.detach(), sample_rate

for sample_rate in [ 2, 4, 8, 16, 32 ]:
    V, sample_rate = inverse_render(sample_rate)
    X, sample_rate = train_nsc(V, sample_rate)

import shutil

# Save the results
results = 'output'

model_file = os.path.join(results, 'model.bin')
complexes_file = os.path.join(results, 'complexes.bin')
corner_points_file = os.path.join(results, 'points.bin')
corner_encodings_file = os.path.join(results, 'encodings.bin')

if os.path.exists(results):
    shutil.rmtree(results)

os.makedirs(results)

torch.save(nsc, model_file)
torch.save(complexes, complexes_file)
torch.save(points, corner_points_file)
torch.save(encodings, corner_encodings_file)

nsc.serialize(complexes, points, encodings, results + '/serialized.nsc')

shutil.copyfile('ref.obj', os.path.join(results, 'ref.obj'))

# TODO: save graphs as well (and use a config file to run all the experiments)
