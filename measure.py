import torch
import trimesh
import configparser

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes

from models import *

file = 'config.ini'

config = configparser.ConfigParser()
config.read(file)

print(config.sections())

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

for section in config.sections():
    print('Measuring benchmark:', section)

    # Needs a 'directory' key
    directory = config[section]['directory']

    # Extract ref.obj
    ref_path = directory + '/ref.obj'
    ref_mesh = trimesh.load(ref_path)
    ref_vertices = torch.tensor(ref_mesh.vertices, dtype=torch.float32).cuda()
    ref_faces = torch.tensor(ref_mesh.faces, dtype=torch.int32).cuda()
    print('Loaded reference model with {} vertices and {} faces'.format(ref_vertices.shape[0], ref_faces.shape[0]))

    # Normalize all measurements; scale to unit cube
    ref_min = torch.min(ref_vertices, dim=0)[0]
    ref_max = torch.max(ref_vertices, dim=0)[0]
    ref_extent = ref_max - ref_min

    ref_vertices = (ref_vertices - ref_min) / ref_extent

    # Load the model and more data
    model, C, P, E = load_binaries(directory)
    print('Loaded neural subdivision complexes with {} complexes and {} vertices'.format(C.shape[0], P.shape[0]), end='\n\n')

    # Evaluate and measure the model at various resolutions
    for resolution in [ 2, 4, 8, 16, 32 ]:
        print('Evaluating at resolution:', resolution)

        eval_vertices, eval_indices = eval(model, C, P, E, sample_rate=resolution)
        print('  > Evaluated NSC mesh with {} vertices and {} faces'.format(eval_vertices.shape[0], eval_indices.shape[0]))

        eval_vertices = (eval_vertices - ref_min) / ref_extent

        # Compute chamfer distance between meshes
        # TODO: separate function
        mref = Meshes(verts=[ ref_vertices ], faces=[ ref_faces ])
        mnsc = Meshes(verts=[ eval_vertices ], faces=[ eval_indices ])

        vertex_loss, normal_loss = chamfer_distance(
            x=mref.verts_padded(),
            y=mnsc.verts_padded(),
            x_normals=mref.verts_normals_padded(),
            y_normals=mnsc.verts_normals_padded(),
            abs_cosine=False
        )

        print('  > Chamfer distance: vertex loss = {:5f}, normal loss = {:5f}'.format(vertex_loss, normal_loss))

    # TODO: run comparisons with decimation