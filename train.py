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

from scripts.render import NVDRenderer

renderer = NVDRenderer(scene_parameters, shading=True, boost=3)

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
for i in range(min(len(renderer.bgs), 5)):
    plt.subplot(1, min(len(renderer.bgs), 5), i + 1)
    plt.imshow(renderer.bgs[i,...,:-1].cpu().numpy(), origin='lower')
    plt.axis('off')

plt.show()

print('Renderer initialized')

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
