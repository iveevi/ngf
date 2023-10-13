import argparse
import meshio
import os
import torch

from torch.utils.cpp_extension import load

# Arguments
parser = argparse.ArgumentParser(description='neural subdivision complexes: measure results')
parser.add_argument('--reference', type=str, help='reference mesh')
parser.add_argument('--sources', type=str, nargs='+', help='source meshes')

args = parser.parse_args()

# Load all necessary extensions
casdf = load(name='casdf',
        sources=[ '../ext/casdf.cu' ],
        extra_include_paths=[ '../glm' ],
        build_directory='../build')

print('Loaded casdf extension')

# Load all meshes
target_path = os.path.basename(args.reference)
source_paths = [ os.path.basename(source) for source in args.sources ]
print('Loaded target mesh: %s' % target_path)
print('Loaded source meshes: %s' % source_paths)

target = meshio.read(args.reference)
sources = [ meshio.read(source) for source in args.sources ]

# Create acceleration structures for each
convert = lambda M: (torch.from_numpy(M.points).float().cuda(), torch.from_numpy(M.cells_dict['triangle']).int().cuda())

target = convert(target)
sources = [ convert(source) for source in sources ]

target = casdf.geometry(target[0].cpu(), target[1].cpu())
source = [ casdf.geometry(source[0].cpu(), source[1].cpu()) for source in sources ]

target_cas = casdf.cas_grid(target, 32)
source_cas = [ casdf.cas_grid(cas, 32) for cas in source ]

target = target.torched()
sources = [ cas.torched() for cas in source ]

# Method to evaluate error
def sample_surface(V, N, T, count=1000):
    # Sample random triangles and baricentric coordinates
    rT = torch.randint(0, T.shape[0], (count,)).cuda()
    rB = torch.rand((count, 2)).cuda()
    rBu, rBv = rB[:, 0], rB[:, 1]
    rBu_sqrt = rBu.sqrt()
    w0 = 1.0 - rBu_sqrt
    w1 = rBu_sqrt * (1.0 - rBv)
    w2 = rBu_sqrt * rBv
    rB = torch.stack([w0, w1, w2], dim=-1)

    Vs = V[T[rT, 0]] * rB[:, 0].unsqueeze(-1) + \
            V[T[rT, 1]] * rB[:, 1].unsqueeze(-1) + \
            V[T[rT, 2]] * rB[:, 2].unsqueeze(-1)

    Ns = N[T[rT, 0]] * rB[:, 0].unsqueeze(-1) + \
            N[T[rT, 1]] * rB[:, 1].unsqueeze(-1) + \
            N[T[rT, 2]] * rB[:, 2].unsqueeze(-1)

    return Vs, torch.nn.functional.normalize(Ns, dim=-1)

def lerped_normals(N, T, B):
    assert T.shape[0] == B.shape[0]

    N0 = N[T[:, 0]]
    N1 = N[T[:, 1]]
    N2 = N[T[:, 2]]

    N = B[:, 0].unsqueeze(-1) * N0 + \
            B[:, 1].unsqueeze(-1) * N1 + \
            B[:, 2].unsqueeze(-1) * N2

    return torch.nn.functional.normalize(N, dim=-1)

def sampled_measurements(target, target_cas, source, source_cas):
    from tqdm import trange

    dpm = 0
    dnormal = 0

    batch = 10_000
    total = 1_000_000

    closest = torch.zeros((batch, 3)).cuda()
    bary    = torch.zeros((batch, 3)).cuda()
    dist    = torch.zeros(batch).cuda()
    index   = torch.zeros(batch, dtype=torch.int32).cuda()

    # TODO: normal vector...

    for i in trange(total // batch):
        S_target, N_target = sample_surface(target[0], target[1], target[2], count=batch)
        S_source, N_source = sample_surface(source[0], source[1], source[2], count=batch)

        rate = source_cas.precache_query_device(S_target)
        source_cas.precache_device()
        source_cas.query_device(S_target, closest, bary, dist, index)
        N_closest = lerped_normals(source[1], source[2][index], bary)

        d0 = torch.sum(torch.linalg.norm(closest - S_target, dim=1))

        dn0 = torch.clamp((N_target * N_closest).sum(dim=1), min=-1, max=1).abs()
        dn0 = torch.acos(dn0)
        dn0 = dn0 * 180 / 3.14159265358979323846
        dn0 = torch.sum(dn0)

        rate = target_cas.precache_query_device(S_source)
        target_cas.precache_device()
        target_cas.query_device(S_source, closest, bary, dist, index)
        N_closest = lerped_normals(target[1], target[2][index], bary)

        d1 = torch.sum(torch.linalg.norm(closest - S_source, dim=1))

        dn1 = torch.clamp((N_source * N_closest).sum(dim=1), min=-1, max=1).abs()
        dn1 = torch.acos(dn1)
        dn1 = dn1 * 180 / 3.14159265358979323846
        dn1 = torch.sum(dn1)

        dpm += (d0 + d1)/(2 * total)
        dnormal += (dn0 + dn1)/(2 * total)

    return dpm.item(), dnormal.item()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
plt.figure(figsize=(16, 9))

min_dpm, max_dpm = 1e9, -1e9
min_dnormal, max_dnormal = 1e9, -1e9

for i in range(len(sources)):
    stem = os.path.splitext(os.path.basename(args.sources[i]))[0].replace('_mesh', '')
    print('Evaluating %s' % stem)
    stem = stem.split('-')
    stem = [ s[0] for s in stem ]
    stem = ''.join(stem)
    print('Stem: %s' % stem)

    dpm, dnormal = sampled_measurements(target, target_cas, sources[i], source_cas[i])
    print('DPM[%s]: %f' % (source_paths[i], dpm))
    print('DNormal[%s]: %f' % (source_paths[i], dnormal))

    plt.scatter(dpm, dnormal, label=stem)
    plt.annotate(stem, xy=(dpm, dnormal),
         textcoords='offset points', ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
         arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=1'))

    min_dpm = min(min_dpm, dpm)
    max_dpm = max(max_dpm, dpm)
    min_dnormal = min(min_dnormal, dnormal)
    max_dnormal = max(max_dnormal, dnormal)

plt.xlabel('DPM')
plt.ylabel('DNormal')
plt.tight_layout()

plt.savefig('measurements.png')
plt.show()
