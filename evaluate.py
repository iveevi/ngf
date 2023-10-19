import argparse
import json
import meshio
import os
import torch

from torch.utils.cpp_extension import load

# Arguments
parser = argparse.ArgumentParser(description='neural subdivision complexes: measure results')
parser.add_argument('--reference', type=str, help='reference mesh')
parser.add_argument('--source', type=str, help='source mesh')
parser.add_argument('--db', type=str, help='database file (json)')

args = parser.parse_args()

assert args.reference is not None
assert args.source is not None
assert args.db is not None

# Expect specific naming conventions for reference and db
assert os.path.basename(args.reference) == 'target.obj'
assert os.path.splitext(args.db)[1] == '.json'

# Load all necessary extensions
casdf = load(name='casdf',
        sources=[ 'ext/casdf.cu' ],
        extra_include_paths=[ 'glm' ],
        build_directory='build')

print('Loaded casdf extension')

# Load all meshes
target_path = os.path.basename(args.reference)
source_path = os.path.basename(args.source)

print('Loaded target mesh: %s' % target_path)
print('Loaded source meshes: %s' % source_path)

target = meshio.read(args.reference)
source = meshio.read(args.source)

# Create acceleration structures for each
convert = lambda M: (torch.from_numpy(M.points).float().cuda(), torch.from_numpy(M.cells_dict['triangle']).int().cuda())

target = convert(target)
source = convert(source)

target = casdf.geometry(target[0].cpu(), target[1].cpu())
source = casdf.geometry(source[0].cpu(), source[1].cpu())

target_cas = casdf.cas_grid(target, 32)
source_cas = casdf.cas_grid(source, 32)

target = target.torched()
source = source.torched()

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

# Perform measurements
# TODO: also run with simplification...
# TODO: handle globs...
# TODO: also do size checks...

dpm, dnormal = sampled_measurements(target, target_cas, source, source_cas)
print('DPM: %f' % dpm)
print('DNormal: %f' % dnormal)

# Open or create the database (json)
if not os.path.exists(args.db):
    with open(args.db, 'w') as f:
        json.dump({}, f)

with open(args.db, 'r') as f:
    db = json.load(f)
    print('Loaded database with %d entries' % len(db))

    # Determine the entry to write to

    # get the directory of the reference
    ref_dir = os.path.dirname(args.reference)
    ref_tag = os.path.basename(ref_dir)
    print('Reference directory: %s' % ref_dir)
    print('Reference tag: %s' % ref_tag)

    # Get information on the source
    source_info = os.path.basename(args.source).split('.')[0]
    print('Source info: %s' % source_info)

    source_tags = source_info.split('-')
    print('Source tags: %s' % source_tags)

    # ref_entry = db.get(ref_tag, {})

    # Add the nested entry with all the source tags
    # for tag in source_tags:
    #     ref_entry = ref_entry.get(tag, {})
    ref_entry = {
        'dpm': dpm,
        'dnormal': dnormal
    }

    for tag in reversed(source_tags):
        ref_entry = { tag: ref_entry }

    # Write the entry back
    db[ref_tag] = ref_entry
    print('Wrote entry: %s' % ref_entry)
    print('Wrote database with %d entries' % len(db))
    print('Database: %s' % db)

# Write the database back
with open(args.db, 'w') as f:
    json.dump(db, f, indent=4)
