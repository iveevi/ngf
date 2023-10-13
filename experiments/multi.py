import argparse
import meshio

from tqdm import trange

from mlp import *
from util import *

# Arguments
parser = argparse.ArgumentParser(description='neural subdivision complexes: mutliple complexes')
parser.add_argument('--target', type=str, help='target tensor output')
parser.add_argument('--source', type=str, help='source geometry')
parser.add_argument('--model', type=str, help='model to use')
parser.add_argument('--loss', type=str, help='loss to use')
parser.add_argument('--lerp', type=str, help='lerp to use')
parser.add_argument('--iterations', type=int, default=1000, help='number of iterations')
parser.add_argument('--display', type=bool, default=False, help='display result')
parser.add_argument('--output', type=str, default='output', help='path prefix for outputs')

args = parser.parse_args()

# Load the target object
source = meshio.read(args.source)
target = torch.load(args.target)

# points = source.points
points = torch.from_numpy(source.points).float().cuda()
complexes = torch.from_numpy(source.cells_dict['quad']).int().cuda()
features = torch.randn((points.shape[0], POINT_ENCODING_SIZE), requires_grad=True, device='cuda', dtype=torch.float32)

# Determine the resolution
sample_rate = int(np.sqrt(target.shape[0] / complexes.shape[0]))

I = indices(complexes, sample_rate=sample_rate)

# Compute reference normals
def triangle_normals(V, T):
    v0 = V[T[:, 0]]
    v1 = V[T[:, 1]]
    v2 = V[T[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v0

    return torch.nn.functional.normalize(torch.cross(e0, e1, dim=1), dim=1)

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

def clerp(f=lambda x: x):
    def ftn(X, U, V):
        lp00 = X[:, 0, :].unsqueeze(1) * f(U.unsqueeze(-1)) * f(V.unsqueeze(-1))
        lp01 = X[:, 1, :].unsqueeze(1) * f(1.0 - U.unsqueeze(-1)) * f(V.unsqueeze(-1))
        lp10 = X[:, 3, :].unsqueeze(1) * f(U.unsqueeze(-1)) * f(1.0 - V.unsqueeze(-1))
        lp11 = X[:, 2, :].unsqueeze(1) * f(1.0 - U.unsqueeze(-1)) * f(1.0 - V.unsqueeze(-1))
        return lp00 + lp01 + lp10 + lp11
    return ftn

models = {
        'pos-enc': MLP_Positional_Encoding,
        'onion-enc': MLP_Positional_Onion_Encoding,
        'morlet-enc': MLP_Positional_Morlet_Encoding,
        'feat-enc': MLP_Feature_Sinusoidal_Encoding,
        'feat-morlet-enc': MLP_Feature_Morlet_Encoding,
        'feat-onion-enc': MLP_Feature_Onion_Encoding,
}

losses = {
        'v': lambda V, N, F: V,
        'n': lambda V, N, F: N,
        'vn': lambda V, N, F: V + N,
        'vn10': lambda V, N, F: V + 10.0 * N,
        'vn100': lambda V, N, F: V + 100.0 * N,
        'vnf': lambda V, N, F: V + N + F,
}

lerps = {
        'linear': lambda x: x,
        'sin': lambda x: torch.sin(32.0 * x * np.pi / 2.0),
        'floor': lambda x: torch.floor(32 * x)/32.0,
        'smooth-floor': lambda x: (32.0 * x - torch.sin(32.0 * 2.0 * x * np.pi)/(2.0 * np.pi)) / 32.0,
}

assert args.model in models
assert args.loss in losses
assert args.lerp in lerps

c = clerp(lerps[args.lerp])
m = models[args.model]().cuda()
l = losses[args.loss]

opt = torch.optim.Adam(list(m.parameters()) + [ features ], lr=1e-3)

LP, LE, UV = sample(complexes, points, features, sample_rate, kernel=c)
V = m(points=LP, encodings=LE, uv=UV)
Vinit = V.detach()

target_normals = triangle_normals(target, I)

history = {}
for i in trange(args.iterations):
    LP, LE, UV = sample(complexes, points, features, sample_rate, kernel=c)
    V = m(points=LP, encodings=LE, uv=UV)

    Vn = triangle_normals(V, I)
    aell = average_edge_length(V, I)

    vertex_loss = (V - target).square().mean()
    normal_loss = aell * (Vn - target_normals).square().mean()
    feature_loss = torch.exp(-torch.cdist(features, features).mean())
    loss = l(vertex_loss, normal_loss, feature_loss)

    history.setdefault('vertex', []).append(vertex_loss.item())
    history.setdefault('normal', []).append(normal_loss.item())
    history.setdefault('feature', []).append(feature_loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()

# Saving results
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
dir = os.path.dirname(args.output)
os.makedirs(dir, exist_ok=True)

# Save result mesh
V = m(points=LP, encodings=LE, uv=UV)
V = V.detach().cpu().numpy()
V_I = shorted_indices(V, complexes, sample_rate)
meshio.write(args.output + '_mesh.obj', meshio.Mesh(V, { 'triangle': V_I }))

# Save loss plot
sns.set_style('darkgrid')

plt.plot(history['vertex'], label='vertex')
plt.plot(history['normal'], label='normal')
plt.plot(history['feature'], label='feature')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.savefig('history.png')
plt.savefig(args.output + '_history.png')

# Save model
results = args.output + '_results'
model_file = os.path.join(results, 'model.bin')
complexes_file = os.path.join(results, 'complexes.bin')
points_file = os.path.join(results, 'points.bin')
features_file = os.path.join(results, 'encodings.bin')

os.makedirs(results, exist_ok=True)
torch.save(m, model_file)
torch.save(complexes, complexes_file)
torch.save(points, points_file)
torch.save(features, features_file)

if args.display:
    import polyscope as ps

    ps.init()

    C_colors = color_code_complexes(complexes, sample_rate)

    target = target.detach().cpu()
    target_I = shorted_indices(target, complexes, sample_rate)
    ps.register_surface_mesh('target', target, target_I).add_color_quantity('complexes', C_colors, defined_on='faces')

    ps.register_surface_mesh('model', V, V_I).add_color_quantity('complexes', C_colors, defined_on='faces')

    Vinit = Vinit.detach().cpu()
    Vinit_I = shorted_indices(Vinit, complexes, sample_rate)
    ps.register_surface_mesh('model-initial', Vinit, Vinit_I).add_color_quantity('complexes', C_colors, defined_on='faces')

    ps.show()
