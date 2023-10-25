import matplotlib.pyplot as plt
import os
import seaborn as sns
import sys
import torch

from torch.utils.cpp_extension import load
from tqdm import trange

from mlp import *
from util import *
from configurations import *
from scripts.geometry import compute_face_normals, compute_vertex_normals

assert len(sys.argv) >= 5, 'Usage: python train.py <directory> <model> <kernel> <iterations> <display=no>'
if not os.path.exists('build'):
    os.makedirs('build')

geometry = load(name="geom_cpp",
        sources=[ "ext/geometry.cpp" ],
        extra_include_paths=[ "glm" ],
        build_directory="build")

directory  = sys.argv[1]
data       = torch.load(directory + '/proxy.pt')
iterations = int(sys.argv[4])
display    = len(sys.argv) == 6 and sys.argv[5] == 'yes'

V         = data['proxy']
complexes = data['complexes']
points    = data['points']

assert V.shape[0] == complexes.shape[0]
sample_rate = V.shape[1]

print('V', V.shape)
print('complexes', complexes.shape)
print('points', points.shape)

print('V', sample_rate, 'complexes', complexes.shape[0], 'points', points.shape[0])

features = torch.randn((points.shape[0], POINT_ENCODING_SIZE), dtype=torch.float32, device='cuda')
print('features', features.shape)
V = V.reshape(-1, 3)

def lerp(X, U, V):
    lp00 = X[:, 0, :].unsqueeze(1) * U.unsqueeze(-1) * V.unsqueeze(-1)
    lp01 = X[:, 1, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * V.unsqueeze(-1)
    lp10 = X[:, 3, :].unsqueeze(1) * U.unsqueeze(-1) * (1.0 - V.unsqueeze(-1))
    lp11 = X[:, 2, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * (1.0 - V.unsqueeze(-1))
    return lp00 + lp01 + lp10 + lp11

def sample(sample_rate, kernel=lerp):
    U = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    V = torch.linspace(0.0, 1.0, steps=sample_rate).cuda()
    U, V = torch.meshgrid(U, V, indexing='ij')

    corner_points = points[complexes, :]
    corner_features = features[complexes, :]

    U, V = U.reshape(-1), V.reshape(-1)
    U = U.repeat((complexes.shape[0], 1))
    V = V.repeat((complexes.shape[0], 1))

    lerped_points = lerp(corner_points, U, V).reshape(-1, 3)
    lerped_features = kernel(corner_features, U, V).reshape(-1, POINT_ENCODING_SIZE)

    return lerped_points, lerped_features

def indices(sample_rate):
    triangles = []
    for i in range(sample_rate - 1):
        for j in range(sample_rate - 1):
            a = i * sample_rate + j
            c = (i + 1) * sample_rate + j
            b, d = a + 1, c + 1
            triangles.append([a, b, c])
            triangles.append([b, d, c])

    return np.array(triangles)

def sample_rate_indices(sample_rate):
    tri_indices = []
    for i in range(complexes.shape[0]):
        ind = indices(sample_rate)
        ind += i * sample_rate ** 2
        tri_indices.append(ind)

    tri_indices = np.concatenate(tri_indices, axis=0)
    tri_indices_tensor = torch.from_numpy(tri_indices).int().cuda()
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

def clerp(f=lambda x: x):
    def ftn(X, U, V):
        lp00 = X[:, 0, :].unsqueeze(1) * f(U.unsqueeze(-1)) * f(V.unsqueeze(-1))
        lp01 = X[:, 1, :].unsqueeze(1) * f(1.0 - U.unsqueeze(-1)) * f(V.unsqueeze(-1))
        lp10 = X[:, 3, :].unsqueeze(1) * f(U.unsqueeze(-1)) * f(1.0 - V.unsqueeze(-1))
        lp11 = X[:, 2, :].unsqueeze(1) * f(1.0 - U.unsqueeze(-1)) * f(1.0 - V.unsqueeze(-1))
        return lp00 + lp01 + lp10 + lp11
    return ftn

m = models[sys.argv[2]]().cuda()
c = clerp(lerps[sys.argv[3]])

optimizer = torch.optim.Adam(list(m.parameters()) + [ features ], lr=1e-3)

I = sample_rate_indices(sample_rate)

base, _ = sample(sample_rate)
cmap    = make_cmap(complexes, base, sample_rate)
remap   = geometry.generate_remapper(complexes.cpu(), cmap, base.shape[0], sample_rate)
F       = remap.remap(I.cpu()).cuda()

VFn = compute_face_normals(V, F)
Vn = compute_vertex_normals(V, F, VFn)
aell = average_edge_length(V, F).item()
print('average edge length', aell)

history = {}
for _ in trange(iterations):
    lerped_points, lerped_features = sample(sample_rate)
    V_pred = m(points=lerped_points, features=lerped_features)

    VF_n_pred = compute_face_normals(V_pred, F)
    V_n_pred = compute_vertex_normals(V_pred, F, VF_n_pred)

    vertex_loss = (V_pred - V).square().mean()
    normal_loss = (V_n_pred - Vn).square().mean()
    loss = vertex_loss + aell * normal_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    history.setdefault('vertex loss', []).append(vertex_loss.item())
    history.setdefault('normal loss', []).append(normal_loss.item())
    history.setdefault('loss', []).append(loss.item())

if display:
    import polyscope as ps

    V_pred = V_pred.detach().cpu().numpy()
    V = V.detach().cpu().numpy()

    I_pred = shorted_indices(V_pred, complexes, sample_rate)
    I = shorted_indices(V, complexes, sample_rate)

    ps.init()
    ps.register_surface_mesh('mesh', V_pred, I_pred)
    ps.register_surface_mesh('target', V, I)
    ps.show()

# Plot losses
sns.set_style('darkgrid')
plt.plot(history['vertex loss'], label='vertex loss')
plt.plot(history['normal loss'], label='normal loss')
plt.plot(history['loss'], label='loss')
plt.yscale('log')
plt.legend()
# plt.show()

# TODO: write all data to the results/models-X...
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
