import configparser
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
import trimesh

from torch.utils.cpp_extension import load

from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds

from tqdm import trange

from models import *

file = 'config.ini'

config = configparser.ConfigParser()
config.read(file)

print('Running measurements for', ' '.join(config.sections()))

# Create a directory for exporting measurements
mdir = 'measurements'
if not os.path.exists(mdir):
    os.makedirs(mdir)

# Compile necessary Torch plugins
if not os.path.exists('build'):
    os.makedirs('build')

casdf = load(name="casdf",
        sources=[ "ext/casdf.cu" ],
        extra_include_paths=[ "glm" ],
        build_directory="build",
)

# Helper functions
def load_binaries(data_dir):
    model = data_dir + '/model.bin'
    model = torch.load(model)

    complexes = data_dir + '/complexes.bin'
    complexes = torch.load(complexes)

    points = data_dir + '/points.bin'
    points = torch.load(points)

    encodings = data_dir + '/encodings.bin'
    encodings = torch.load(encodings)

    return model, complexes, points, encodings

def lerp(X, U, V):
    lp00 = X[:, 0, :].unsqueeze(1) * U.unsqueeze(-1) * V.unsqueeze(-1)
    lp01 = X[:, 1, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * V.unsqueeze(-1)
    lp10 = X[:, 3, :].unsqueeze(1) * U.unsqueeze(-1) * (1.0 - V.unsqueeze(-1))
    lp11 = X[:, 2, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * (1.0 - V.unsqueeze(-1))
    return lp00 + lp01 + lp10 + lp11

def eval(model, complexes, points, encodings, sample_rate=16):
    U = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    V = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    U, V = torch.meshgrid(U, V)

    corner_points = points[complexes, :]
    corner_encodings = encodings[complexes, :]

    U, V = U.reshape(-1), V.reshape(-1)
    U = U.repeat((complexes.shape[0], 1))
    V = V.repeat((complexes.shape[0], 1))

    LP = lerp(corner_points, U, V).reshape(-1, 3)
    LE = lerp(corner_encodings, U, V).reshape(-1, POINT_ENCODING_SIZE)


    all_indices = []
    for i in range(complexes.shape[0]):
        ind = indices(sample_rate)
        ind += i * sample_rate ** 2
        all_indices.append(ind)

    all_indices = np.concatenate(all_indices, axis=0)
    all_indices = torch.from_numpy(all_indices).cuda()

    return model(LP, LE), all_indices

def sampled_measurements(mref, cas_ref, mnsc):
    dpm = 0

    batch = 10_000
    total = 1_000_000

    closest = torch.zeros((batch, 3))
    bary = torch.zeros((batch, 3))
    dist = torch.zeros(batch)
    index = torch.zeros(batch, dtype=torch.int32)
        
    cas_nsc = casdf.geometry(
        mnsc.verts_packed().float().cpu(),
        mnsc.verts_normals_packed().float().cpu(),
        mnsc.faces_packed().int().cpu())
    
    cas_nsc = casdf.cas_grid(cas_nsc, 64)

    for i in trange(total // batch):
        sampled_nsc = sample_points_from_meshes(mnsc, batch)[0].cpu()
        rate = cas_ref.precache_query(sampled_nsc)
        cas_ref.query(sampled_nsc, closest, bary, dist, index)
        d = torch.sum(torch.linalg.norm(closest - sampled_nsc, dim=1))

        sampled_ref = sample_points_from_meshes(mref, batch)[0].cpu()
        rate = cas_nsc.precache_query(sampled_ref)
        cas_nsc.query(sampled_ref, closest, bary, dist, index)
        d += torch.sum(torch.linalg.norm(closest - sampled_ref, dim=1))

        dpm += d/(2 * total)

    return dpm

for section in config.sections():
    print('\n' + '-' * 40)
    print(' ' * 10, 'BENCHMARKING', section)
    print('-' * 40)

    # Needs a 'directory' key
    directory = config[section]['directory']

    # Extract ref.obj
    ref_path = directory + '/ref.obj'
    ref_mesh = trimesh.load(ref_path)
    ref_vertices = torch.tensor(ref_mesh.vertices, dtype=torch.float32).cuda()
    ref_faces = torch.tensor(ref_mesh.faces, dtype=torch.int32).cuda()
    print('Loaded reference model with {} vertices and {} faces'.format(ref_vertices.shape[0], ref_faces.shape[0]))

    # Acceleration structure for the reference mesh
    mref = Meshes(verts=[ ref_vertices ], faces=[ ref_faces ])

    cas_ref = casdf.geometry(
        mref.verts_packed().float().cpu(),
        mref.verts_normals_packed().float().cpu(),
        mref.faces_packed().int().cpu())
    
    print(' cas_Ref:', cas_ref)
    
    cas_ref = casdf.cas_grid(cas_ref, 64)

    print(' cas_Ref:', cas_ref)

    # Load the model and more data
    model, C, P, E = load_binaries(directory)
    print('Loaded neural subdivision complexes with {} complexes and {} vertices'.format(C.shape[0], P.shape[0]))

    # Evaluate and measure the model at various resolutions
    vertex_losses = []
    normal_losses = []

    dpms = []
    dnormals = []

    for resolution in [ 2, 4, 8, 16, 32 ]:
        print('\nEvaluating at resolution:', resolution)

        eval_vertices, eval_indices = eval(model, C, P, E, sample_rate=resolution)
        print('  > Evaluated NSC mesh with {} vertices and {} faces'.format(eval_vertices.shape[0], eval_indices.shape[0]))

        # eval_vertices = (eval_vertices - ref_min.cuda()) / ref_extent.cuda()

        # Compute chamfer distance between meshes
        # TODO: separate function
        mnsc = Meshes(verts=[ eval_vertices ], faces=[ eval_indices ])

        vertex_loss, normal_loss = chamfer_distance(
            x=mref.verts_padded(),
            y=mnsc.verts_padded(),
            x_normals=mref.verts_normals_padded(),
            y_normals=mnsc.verts_normals_padded(),
            abs_cosine=False
        )

        print('  > Chamfer distance: vertex loss = {:5f}, normal loss = {:5f}'.format(vertex_loss, normal_loss))

        vertex_losses.append(vertex_loss.item())
        normal_losses.append(normal_loss.item())

        dpm = sampled_measurements(mref, cas_ref, mnsc)
        print('  > Average point-mesh distance:', dpm)
        dpms.append(dpm.item())

    # Plot losses
    sns.set_theme(style='darkgrid')

    plt.figure(figsize=(20, 15))
    plt.plot([ 2, 4, 8, 16, 32 ], vertex_losses, label='Chamfer loss')
    # plt.plot([ 2, 4, 8, 16, 32 ], normal_losses, label='Normal loss')
    plt.plot([ 2, 4, 8, 16, 32 ], dpms, label='Point-mesh distance')
    plt.yscale('log')
    plt.xlabel('Resolution')
    plt.ylabel('Losses')
    plt.legend()
    plt.savefig(os.path.join(mdir, section + '.png'))

    # TODO: comparison with binary sizes... (KB and log?)

    # TODO: sampled losses...

    # TODO: run comparisons with decimation

    # TODO: save into a format that another library can preview (custom...)