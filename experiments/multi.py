import argparse
import meshio

from tqdm import trange

from mlp import *
from util import *

# Arguments
parser = argparse.ArgumentParser(description='neural subdivision complexes: mutliple complexes')
parser.add_argument('--target', type=str, help='target tensor output')
parser.add_argument('--source', type=str, help='source geometry')

args = parser.parse_args()

# Load the target object
source = meshio.read(args.source)
target = torch.load(args.target)

print('Loaded source mesh', args.source)
print('Loaded target tensor', target.shape)

# points = source.points
points = torch.from_numpy(source.points).float().cuda()
complexes = torch.from_numpy(source.cells_dict['quad']).int().cuda()
features = torch.randn((points.shape[0], POINT_ENCODING_SIZE), requires_grad=True, device='cuda', dtype=torch.float32)

# Determine the resolution
sample_rate = int(np.sqrt(target.shape[0] / complexes.shape[0]))
print('Sample rate:', sample_rate)

I = indices(complexes, sample_rate=sample_rate)

# Compute reference normals
def triangle_normals(points, triangles):
    v0 = points[triangles[:, 0]]
    v1 = points[triangles[:, 1]]
    v2 = points[triangles[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v0

    return torch.cross(e0, e1, dim=1)

# TODO: try different numbers, and fewer iterations..
# and different losses...

m = MLP_Positional_Encoding().cuda()
opt = torch.optim.Adam(list(m.parameters()) + [ features ], lr=1e-3)

LP, LE, UV = sample(complexes, points, features, sample_rate)
V = m(points=LP, encodings=LE, uv=UV)
Vinit = V.detach()

target_normals = triangle_normals(target, I)

history = {}
for i in trange(1_000):
    LP, LE, UV = sample(complexes, points, features, sample_rate)
    V = m(points=LP, encodings=LE, uv=UV)

    Vn = triangle_normals(V, I)

    vertex_loss = (V - target).square().mean()
    normal_loss = 1e3 * (Vn - target_normals).square().mean()
    loss = vertex_loss + normal_loss

    history.setdefault('vertex', []).append(vertex_loss.item())
    history.setdefault('normal', []).append(normal_loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()

import polyscope as ps

ps.init()

C_colors = color_code_complexes(complexes, sample_rate)

target = target.detach().cpu()
target_I = shorted_indices(target, complexes, sample_rate)
ps.register_surface_mesh('target', target, target_I).add_color_quantity('complexes', C_colors, defined_on='faces')

V = V.detach().cpu()
V_I = shorted_indices(V, complexes, sample_rate)
ps.register_surface_mesh('model', V, V_I).add_color_quantity('complexes', C_colors, defined_on='faces')

Vinit = Vinit.detach().cpu()
Vinit_I = shorted_indices(Vinit, complexes, sample_rate)
ps.register_surface_mesh('model-initial', Vinit, Vinit_I).add_color_quantity('complexes', C_colors, defined_on='faces')

ps.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

plt.plot(history['vertex'], label='vertex')
plt.plot(history['normal'], label='normal')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.savefig('history.png')
