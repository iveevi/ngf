import argparse
import json
import meshio
import os
import sys
import torch

from torch.utils.cpp_extension import load
from scripts.geometry import compute_face_normals, compute_vertex_normals

from util import *

# Arguments
assert len(sys.argv) == 4, 'evaluate.py <reference> <directory> <db>'

reference = sys.argv[1]
directory = sys.argv[2]
db = sys.argv[3]

assert reference is not None
assert directory is not None
assert db is not None

# Expect specific naming conventions for reference and db
assert os.path.basename(reference) == 'target.obj'
assert os.path.splitext(db)[1] == '.json'

# Load all necessary extensions
casdf = load(name='casdf',
        sources=[ 'ext/casdf.cu' ],
        extra_include_paths=[ 'glm' ],
        build_directory='build')

geom_cpp = load(name="geom_cpp",
        sources=[ "ext/geometry.cpp" ],
        extra_include_paths=[ "glm" ],
        build_directory="build")

print('Loaded all extensions')

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

# Writing to the database
def write(catag, dpm, dnormal, size=0):
    global db

    # Open or create the database (json)
    if not os.path.exists(db):
        with open(db, 'w') as f:
            json.dump({}, f)

    with open(db, 'r') as f:
        db_json = json.load(f)
        # print('Loaded database with %d entries' % len(db_json))

        # Determine the entry to write to

        # get the directory of the reference
        ref_dir = os.path.dirname(reference)
        ref_tag = os.path.basename(ref_dir)
        # print('Reference directory: %s' % ref_dir)
        # print('Reference tag: %s' % ref_tag)

        # Get information on the source
        tags = catag.split('-')
        print('Source tags: %s' % tags)

        # Add the nested entry with all the source tags
        ref_entry = {
            'dpm': dpm,
            'dnormal': dnormal,
            'size': size
        }

        for tag in reversed(tags):
            ref_entry = { tag: ref_entry }

        # Write the entry back
        print('Writing entry: %s' % ref_entry)
        if ref_tag in db_json:
            # print('existing: %s' % db_json[ref_tag])
            # db_json[ref_tag].update(ref_entry)
            db_tmp = db_json[ref_tag]
            while True:
                # print('  attempting to update: %s with %s' % (db_tmp, ref_entry))
                key = list(ref_entry.keys())[0]
                # print('  key: %s' % key)
                if key in db_tmp:
                    db_tmp = db_tmp[key]
                    ref_entry = ref_entry[key]
                else:
                    db_tmp[key] = ref_entry[key]
                    break
        else:
            db_json[ref_tag] = ref_entry

        # print('Wrote database with %d entries' % len(db_json))
        # print('Database: %s' % db_json)

    # Write the database back
    with open(db, 'w') as f:
        json.dump(db_json, f, indent=4)

# Converting meshio to torch
convert = lambda M: (torch.from_numpy(M.points).float().cuda(), torch.from_numpy(M.cells_dict['triangle']).int().cuda())

# Erase existing directory data from the database
if not os.path.exists(db):
    with open(db, 'w') as f:
        json.dump({}, f)
else:
    with open(db, 'r') as f:
        db_json = json.load(f)
        print('Loaded database with %d entries' % len(db_json))

        # Determine the entry to write to

        # get the directory of the reference
        ref_dir = os.path.dirname(reference)
        ref_tag = os.path.basename(ref_dir)
        print('Reference directory: %s' % ref_dir)
        print('Reference tag: %s' % ref_tag)

        if ref_tag in db_json:
            print('Erasing existing entry: %s' % db_json[ref_tag])
            del db_json[ref_tag]

    # Write the database back
    with open(db, 'w') as f:
        json.dump(db_json, f, indent=4)

# Load target reference meshe
print('Loading target mesh: %s' % reference)

target_path = os.path.basename(reference)
target = meshio.read(reference)
target = convert(target)

target = casdf.geometry(target[0].cpu(), target[1].cpu())
target_cas = casdf.cas_grid(target, 32)
target = target.torched()

# Recursively find all simple meshes and model configurations
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.obj'):
            path = os.path.join(root, file)
            print('Loading source mesh: %s' % path)

            source = meshio.read(path)
            source = convert(source)

            fn = compute_face_normals(source[0], source[1])
            vn = compute_vertex_normals(source[0], source[1], fn)

            source = casdf.geometry(source[0].cpu(), vn.cpu(), source[1].cpu())
            source_cas = casdf.cas_grid(source, 32)
            source = source.torched()

            # Compute raw binary byte size
            vertex_bytes = source[0].numel() * source[0].element_size()
            index_bytes = source[2].numel() * source[2].element_size()
            total = vertex_bytes + index_bytes

            dpm, dnormal = sampled_measurements(target, target_cas, source, source_cas)
            print('  > DPM: %f' % dpm)
            print('  > DNormal: %f' % dnormal)
            print('  > Size: %d KB' % (total/1024))

            catag = os.path.basename(file)
            catag = os.path.splitext(catag)[0]
            write(catag, dpm, dnormal, size=total) # TODO: compute the sizes...
    for nsc in dirs:
        print('Loading NSC representation: %s' % nsc)

        data = torch.load(os.path.join(root, nsc, 'model.pt'))

        m = data['model']
        c = data['complexes']
        p = data['points']
        f = data['features']

        print('  > c:', c.shape)
        print('  > p:', p.shape)
        print('  > f:', f.shape)

        lerper = nsc.split('-')[1]
        print('  > lerper: %s' % lerper)

        ker = clerp(lerps[lerper])
        print('  > clerp: %s' % ker)

        # Compute byte size of the representation
        feature_bytes = f.numel() * f.element_size()
        index_bytes = c.numel() * c.element_size()
        model_bytes = sum([ p.numel() * p.element_size() for p in m.parameters() ])
        vertex_bytes = p.numel() * p.element_size()
        total = feature_bytes + index_bytes + model_bytes + vertex_bytes

        print('  > Size: %d KB' % (total/1024))

        for rate in [ 2, 4, 8, 16 ]:
            lerped_points, lerped_features = sample(c, p, f, rate, kernel=ker)
            # print('    > lerped_points:', lerped_points.shape)
            # print('    > lerped_features:', lerped_features.shape)

            cmap = make_cmap(c, p, lerped_points, rate)
            F, _ = geom_cpp.sdc_weld(c.cpu(), cmap, lerped_points.shape[0], rate)
            F = F.cuda()

            vertices = m(points=lerped_points, features=lerped_features)

            # import polyscope as ps
            # ps.init()
            # ps.register_surface_mesh('mesh', lerped_points.detach().cpu().numpy(), F.cpu().numpy())
            # ps.show()

            fn = compute_face_normals(vertices, F)
            vn = compute_vertex_normals(vertices, F, fn)

            if torch.isnan(vertices).any():
                print('  > vertices has nan')
                continue
            if torch.isnan(F).any():
                print('  > F has nan')
                continue
            if torch.isnan(vn).any():
                print('  > vn has nan')
                continue

            source = casdf.geometry(vertices.cpu(), vn.cpu(), F.cpu())
            source_cas = casdf.cas_grid(source, 32)
            source = source.torched()

            dpm, dnormal = sampled_measurements(target, target_cas, source, source_cas)

            catag = nsc + '-r' + str(rate)
            print('  > rate: %d, tag: %s' % (rate, catag))
            print('    > DPM: %f' % dpm)
            print('    > DNormal: %f' % dnormal)

            write(catag, dpm, dnormal, size=total)
