import argparse
import meshio
import os

from tqdm import trange
from torch.utils.cpp_extension import load

from mlp import *
from util import *

# Arguments
parser = argparse.ArgumentParser(description='neural subdivision complexes: mutliple complexes')
parser.add_argument('--target', type=str, help='target geometry')
parser.add_argument('--source', type=str, help='source geometry')
parser.add_argument('--resolution', type=int, help='sample resolution')
parser.add_argument('--model', type=str, help='model to use')
parser.add_argument('--loss', type=str, help='loss to use')
parser.add_argument('--lerp', type=str, help='lerp to use')
parser.add_argument('--iterations', type=int, default=1000, help='number of iterations')
parser.add_argument('--display', type=bool, default=False, help='display result')
parser.add_argument('--output', type=str, default='output', help='path prefix for outputs')

args = parser.parse_args()

# Load all necessary extensions
if not os.path.exists('../build'):
    os.makedirs('../build')

geom = load(name='geom',
        sources=[ '../ext/geometry.cpp' ],
        extra_include_paths=[ '../glm' ],
        build_directory='../build')

print('Loaded geometry extension')

# TODO: make geom obselete; use a new file instead

casdf = load(name='casdf',
        sources=[ '../ext/casdf.cu' ],
        extra_include_paths=[ '../glm' ],
        build_directory='../build')

print('Loaded casdf extension')

# Load the target object
target = meshio.read(args.target)
source = meshio.read(args.source)

# points = source.points
points = torch.from_numpy(source.points).float().cuda()
complexes = torch.from_numpy(source.cells_dict['quad']).int().cuda()
features = torch.randn((points.shape[0], POINT_ENCODING_SIZE), requires_grad=True, device='cuda', dtype=torch.float32)

# Determine the resolution
I = indices(complexes, sample_rate=args.resolution)

# Compute reference normals
def triangle_normls(V, T):
    v0 = V[T[:, 0], :]
    v1 = V[T[:, 1], :]
    v2 = V[T[:, 2], :]

    v01 = v1 - v0
    v02 = v2 - v0

    return torch.cross(v01, v02, dim=1)

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
        'cubic': lambda x: 25 * x ** 3/3.0 - 25 * x ** 2 + 31 * x/6.0,
}

assert args.model in models
assert args.loss in losses
assert args.lerp in lerps

c = clerp(lerps[args.lerp])
m = models[args.model]().cuda()
l = losses[args.loss]

opt = torch.optim.Adam(list(m.parameters()) + [ features ], lr=1e-3)

tch_target_vertices = torch.from_numpy(target.points).float().cuda()
tch_target_triangles = torch.from_numpy(target.cells_dict['triangle']).int().cuda()

cas = casdf.geometry(tch_target_vertices.cpu(), tch_target_triangles.cpu())
cas = casdf.cas_grid(cas, 32)

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

def sample_surface(V, T, count=1000):
    # Sample random triangles and baricentric coordinates
    rT = torch.randint(0, T.shape[0], (count,)).cuda()
    rB = torch.rand((count, 2)).cuda()
    rBu, rBv = rB[:, 0], rB[:, 1]
    rBu_sqrt = rBu.sqrt()
    w0 = 1.0 - rBu_sqrt
    w1 = rBu_sqrt * (1.0 - rBv)
    w2 = rBu_sqrt * rBv
    rB = torch.stack([w0, w1, w2], dim=-1)
    return V[T[rT, 0]] * rB[:, 0].unsqueeze(-1) + \
            V[T[rT, 1]] * rB[:, 1].unsqueeze(-1) + \
            V[T[rT, 2]] * rB[:, 2].unsqueeze(-1)


import polyscope as ps

ps.init()

for i in range(5):
    LP, LE, UV = sample(complexes, points, features, args.resolution)
    V = m(points=LP, encodings=LE, uv=UV)
    Vinit = V.detach()

    Tv          = None
    if i == 0:
        Tv = LP.clone().requires_grad_(True)
    else:
        Tv = V.detach().clone().requires_grad_(True)

    I_tch       = torch.from_numpy(I).int().cuda()

    cmap        = make_cmap(complexes, LP, args.resolution)
    F, remap    = geom.sdc_weld(complexes.cpu(), cmap, Tv.shape[0], args.resolution)
    F           = F.cuda()
    vgraph      = casdf.vertex_graph(F.cpu())

    Tv_opt      = torch.optim.Adam([Tv], lr=1e-3)

    closest     = torch.zeros((Tv.shape[0], 3)).cuda()
    bary        = torch.zeros((Tv.shape[0], 3)).cuda()
    dist        = torch.zeros(Tv.shape[0]).cuda()
    index       = torch.zeros(Tv.shape[0], dtype=torch.int32).cuda()

    samples     = 10_000
    sample_bary = torch.zeros((samples, 3), dtype=torch.float32).cuda()
    sample_tris = torch.zeros(samples, dtype=torch.int32).cuda()

    history = {}
    for _ in trange(1_000):
        # Direct loss computation
        rate = cas.precache_query_device(Tv)

        cas.precache_device()
        cas.query_device(Tv, closest, bary, dist, index)

        direct_loss = torch.sum((closest - Tv).square())

        # Sampled loss computation
        Vrandom = sample_surface(tch_target_vertices, tch_target_triangles, count=samples)
        casdf.barycentric_closest_points(Tv, F, Vrandom, sample_bary, sample_tris)
        Ts = F[sample_tris]
        Vreconstructed = Tv[Ts[:, 0]] * sample_bary[:, 0].unsqueeze(-1) + \
                            Tv[Ts[:, 1]] * sample_bary[:, 1].unsqueeze(-1) + \
                            Tv[Ts[:, 2]] * sample_bary[:, 2].unsqueeze(-1)

        sampled_loss = torch.sum((Vrandom - Vreconstructed).square())

        # Laplacian loss
        Tv_smoothed = vgraph.smooth_device(Tv, 1.0)
        laplacian_loss = torch.sum((Tv - Tv_smoothed).square())

        # TODO: match normal vectors?

        # TODO: tirangle area min/maxing...
        loss = direct_loss + sampled_loss + laplacian_loss

        # history.setdefault('direct', []).append(direct_loss.item())
        # history.setdefault('sampled', []).append(sampled_loss.item())
        # history.setdefault('laplacian', []).append(laplacian_loss.item())

        Tv_opt.zero_grad()
        loss.backward()
        Tv_opt.step()

    Tv = geom.sdc_separate(Tv.detach().cpu(), remap).cuda()

    I_tch = torch.from_numpy(I).cuda()
    Tvn = compute_face_normals(Tv, I_tch)
    Tvn = compute_vertex_normals(Tv, I_tch, Tvn)
    # Tvn = triangle_normls(Tv, I_tch)

    history = {}
    for _ in trange(args.iterations):
        LP, LE, UV = sample(complexes, points, features, args.resolution, kernel=c)
        V = m(points=LP, encodings=LE, uv=UV)

        # Vn = triangle_normals(V, I)
        # print('V NaNs:', torch.isnan(V).sum().item())
        Fn = compute_face_normals(V, I_tch)
        # print('Fn NaNs:', torch.isnan(Fn).sum().item())
        Vn = compute_vertex_normals(V, I_tch, Fn)
        # print('Vn NaNs:', torch.isnan(Vn).sum().item())
        # Fn = triangle_normls(V, I_tch)
        aell = average_edge_length(V, I)

        vertex_loss = (V - Tv).square().mean()
        # normal_loss = aell * (Vn - Tvn).square().mean()
        normal_loss = (Vn - Tvn).square().mean()
        feature_loss = torch.exp(-torch.cdist(features, features).mean())
        loss = l(vertex_loss, normal_loss, feature_loss)

        history.setdefault('vertex', []).append(vertex_loss.item())
        history.setdefault('normal', []).append(normal_loss.item())
        history.setdefault('feature', []).append(feature_loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    C_colors = color_code_complexes(complexes, args.resolution)
    Tv_sI = shorted_indices(Tv.cpu(), complexes, args.resolution)
    ps.register_surface_mesh(f'Tv{i}', Tv.cpu(), Tv_sI).add_color_quantity('C', C_colors, defined_on='faces')
    V_sI = shorted_indices(V.detach().cpu(), complexes, args.resolution)
    ps.register_surface_mesh(f'V{i}', V.detach().cpu(), V_sI).add_color_quantity('C', C_colors, defined_on='faces')

    ps.show()

ps.show()

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
V_I = shorted_indices(V, complexes, args.resolution)
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

# if args.display:
#     import polyscope as ps
#
#     ps.init()
#
#     C_colors = color_code_complexes(complexes, sample_rate)
#
#     target = target.detach().cpu()
#     target_I = shorted_indices(target, complexes, sample_rate)
#     ps.register_surface_mesh('target', target, target_I).add_color_quantity('complexes', C_colors, defined_on='faces')
#
#     ps.register_surface_mesh('model', V, V_I).add_color_quantity('complexes', C_colors, defined_on='faces')
#
#     Vinit = Vinit.detach().cpu()
#     Vinit_I = shorted_indices(Vinit, complexes, sample_rate)
#     ps.register_surface_mesh('model-initial', Vinit, Vinit_I).add_color_quantity('complexes', C_colors, defined_on='faces')
#
#     ps.show()
