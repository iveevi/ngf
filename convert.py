import os
import polyscope as ps
import polyscope.imgui as imgui
import sys
import torch
import trimesh

from models import *

# Get source directory as first argument
if len(sys.argv) < 2:
    print('Usage: python convert.py <data_dir>')
    sys.exit(1)

data_dir = sys.argv[1]
print('Loading from directory:', data_dir)

total_size = 0
model = data_dir + '/model.bin'
total_size += os.path.getsize(model)
model = torch.load(model)

complexes = data_dir + '/complexes.bin'
total_size += os.path.getsize(complexes)
complexes = torch.load(complexes)

corner_points = data_dir + '/points.bin'
total_size += os.path.getsize(corner_points)
corner_points = torch.load(corner_points)

corner_encodings = data_dir + '/encodings.bin'
total_size += os.path.getsize(corner_encodings)
corner_encodings = torch.load(corner_encodings)

print('Total size:', total_size / 1024 / 1024, 'MB')

l0, l1, l2 = model.encoding_linears
print('l0:', l0.weight.shape, l0.bias.shape)
print('l1:', l1.weight.shape, l1.bias.shape)
print('l2:', l2.weight.shape, l2.bias.shape)

print(complexes.shape, complexes.dtype)
print(corner_points.shape, corner_points.dtype)
print(corner_encodings.shape, corner_encodings.dtype)

# Write all data to file in raw binary format, without torch overhead
name = os.path.basename(data_dir)
print('Writing to:', name + '.nsc')
with open(name + '.nsc', 'wb') as f:
    # Write size information
    f.write(np.array([
        complexes.shape[0],
        corner_points.shape[0],
        ENCODING_SIZE,
        POINT_ENCODING_SIZE,
        MATRIX_SIZE
    ], dtype=np.int32).tobytes())
    
    # Complexes
    complexes_bytes = complexes.cpu().numpy().astype(np.int32).tobytes()
    print('complexes_bytes:', len(complexes_bytes))
    f.write(complexes_bytes)

    # Corner points and encodings
    K = corner_points.shape[0]
    assert(corner_points.shape[0] == K)
    assert(corner_encodings.shape[0] == K)
    corner_points_bytes = corner_points.detach().cpu().numpy().astype(np.float32).tobytes()
    corner_encodings_bytes = corner_encodings.detach().cpu().numpy().astype(np.float32).tobytes()
    print('corner_points_bytes:', len(corner_points_bytes))
    print('corner_encodings_bytes:', len(corner_encodings_bytes))
    f.write(corner_points_bytes)
    f.write(corner_encodings_bytes)

    # Write the model parameters
    W0, H0 = l0.weight.shape
    print('W0, H0:', W0, H0)
    l0_weights = l0.weight.detach().cpu().numpy().astype(np.float32).tobytes()
    l0_bias = l0.bias.detach().cpu().numpy().astype(np.float32).tobytes()
    f.write(np.array([W0, H0], dtype=np.int32).tobytes())
    bytes = f.write(l0_weights)
    print('l0_weights:', bytes)
    bytes = f.write(l0_bias)
    print('l0_bias:', bytes)
    
    W1, H1 = l1.weight.shape
    print('W1, H1:', W1, H1)
    l1_weights = l1.weight.detach().cpu().numpy().astype(np.float32).tobytes()
    l1_bias = l1.bias.detach().cpu().numpy().astype(np.float32).tobytes()
    f.write(np.array([W1, H1], dtype=np.int32).tobytes())
    bytes = f.write(l1_weights)
    print('l1_weights:', bytes)
    bytes = f.write(l1_bias)
    print('l1_bias:', bytes)

    W2, H2 = l2.weight.shape
    print('W2, H2:', W2, H2)
    l2_weights = l2.weight.detach().cpu().numpy().astype(np.float32).tobytes()
    l2_bias = l2.bias.detach().cpu().numpy().astype(np.float32).tobytes()
    f.write(np.array([W2, H2], dtype=np.int32).tobytes())
    bytes = f.write(l2_weights)
    print('l2_weights:', bytes)
    bytes = f.write(l2_bias)
    print('l2_bias:', bytes)
