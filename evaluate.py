import imageio
import json
import meshio
import os
import shutil
import subprocess
import sys
import torch

import optext

# from torch.utils.cpp_extension import load
from scripts.geometry import compute_face_normals, compute_vertex_normals

from util import *
from configurations import *

# Arguments
assert len(sys.argv) == 4, 'evaluate.py <reference> <directory> <db>'
# TODO: directory and /db.sjon

reference = sys.argv[1]
directory = sys.argv[2]
db = sys.argv[3]

assert reference is not None
assert directory is not None
assert db is not None

# Expect specific naming conventions for reference and db
assert os.path.basename(reference) == 'target.obj'
assert os.path.splitext(db)[1] == '.json'

# Converting meshio to torch
convert = lambda M: (torch.from_numpy(M.points).float().cuda(), torch.from_numpy(M.cells_dict['triangle']).int().cuda())

# Load target reference meshe
print('Loading target mesh: %s' % reference)

target_path = os.path.basename(reference)
target = meshio.read(reference)

print('  > %d vertices, %d faces' % (target.points.shape[0], target.cells[0].data.shape[0]))

v_ref = target.points
min = np.min(v_ref, axis=0)
max = np.max(v_ref, axis=0)
center = (min + max) / 2.0
extent = np.linalg.norm(max - min) / 2.0
normalize = lambda x: (x - center) / extent

target.points = normalize(target.points)
target = convert(target)

target = optext.geometry(target[0].cpu(), target[1].cpu())
target_cas = optext.cached_grid(target, 32)
target = target.torched()

# Load scene cameras
# from scripts.load_xml import load_scene
from scripts.render import NVDRenderer

# TODO: put into a util function/script
ref_directory   = os.path.dirname(reference)
simplified      = os.path.join(ref_directory, 'simplified.obj')

simplified_mesh = meshio.read(simplified)
# v_simplified    = torch.from_numpy(simplified_mesh.points).float().cuda()

v_simplified    = normalize(simplified_mesh.points)
v_simplified    = torch.from_numpy(v_simplified).float().cuda()

f_simplified    = torch.from_numpy(simplified_mesh.cells_dict['triangle']).int().cuda()
fn_simplified   = compute_face_normals(v_simplified, f_simplified)
n_simplified    = compute_vertex_normals(v_simplified, f_simplified, fn_simplified)

cameras         = 100
seeds           = list(torch.randint(0, f_simplified.shape[0], (cameras,)).numpy())
target_geometry = optext.geometry(v_simplified.cpu(), n_simplified.cpu(), f_simplified.cpu())
target_geometry = target_geometry.deduplicate()
clusters        = optext.cluster_geometry(target_geometry, seeds, 10)
# FIXME:
# clusters        = optext.cluster_geometry(target_geometry, seeds, 2)

# Compute the centroid and normal for each cluster
cluster_centroids = []
cluster_normals = []

for cluster in clusters:
    faces = f_simplified[cluster]

    v0 = v_simplified[faces[:, 0]]
    v1 = v_simplified[faces[:, 1]]
    v2 = v_simplified[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    centroids = centroids.mean(dim=0)

    normals = torch.cross(v1 - v0, v2 - v0)
    normals = normals.mean(dim=0)
    normals = normals / torch.norm(normals)

    cluster_centroids.append(centroids)
    cluster_normals.append(normals)

cluster_centroids = torch.stack(cluster_centroids, dim=0)
cluster_normals = torch.stack(cluster_normals, dim=0)

canonical_up = torch.tensor([0.0, 1.0, 0.0], device='cuda')
cluster_eyes = cluster_centroids + cluster_normals * 1.0
cluster_ups = torch.stack(len(clusters) * [ canonical_up ], dim=0)
cluster_rights = torch.cross(cluster_normals, cluster_ups)
cluster_ups = torch.cross(cluster_rights, cluster_normals)

all_views = [ lookat(eye, view_point, up) for eye, view_point, up in zip(cluster_eyes, cluster_centroids, cluster_ups) ]

scene = {}
scene['view_mats'] = all_views

# Also load an environment map
# TODO: path/constants file
environment = imageio.imread('media/images/environment.hdr', format='HDR-FI')
environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
alpha       = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
environment = torch.cat((environment, alpha), dim=-1)

scene['envmap']       = environment
scene['envmap_scale'] = 1.0

print('Loaded scene with %d cameras' % len(scene['view_mats']))

# TODO: use higher resolution...
scene['res_x'] = 1920
scene['res_y'] = 1080
scene['fov'] = 45.0
scene['near_clip'] = 0.1
scene['far_clip'] = 1000.0
scene['view_mats'] = torch.stack(scene['view_mats'], dim=0)

renderer = NVDRenderer(scene)

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

    batch = 100_000
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

def alpha_blend(img):
    alpha = img[..., 3:]
    return img[..., :3] * alpha + (1.0 - alpha)

def render_loss(target, source, alt=None, alt_n=None):
    batch = 10
    cameras = scene['view_mats']
    cameras_count = cameras.shape[0]
    losses = []

    tV, tN, tF = target[0], target[1].cuda(), target[2].cuda()
    sV, sN, sF = source[0], source[1].cuda(), source[2].cuda()

    if alt is not None:
        sF = alt

    if alt_n is not None:
        sN = alt_n

    assert cameras_count % batch == 0
    for i in range(0, cameras_count, batch):
        camera_batch = cameras[i:i + batch]

        t_imgs = renderer.render(tV, tN, tF, camera_batch)
        s_imgs = renderer.render(sV, sN, sF, camera_batch)

        t_imgs = alpha_blend(t_imgs)
        s_imgs = alpha_blend(s_imgs)

        loss = torch.mean(torch.abs(t_imgs - s_imgs))
        losses.append(loss.item())

    # Present view
    eye = torch.tensor([ 0, 0, -2.5 ], device='cuda')
    up = torch.tensor([0.0, -1.0, 0.0], device='cuda')
    center = torch.tensor([0.0, 0.0, 0.0], device='cuda')

    camera = lookat(eye, center, up).unsqueeze(0)

    t_img = renderer.render(tV, tN, tF, camera)[0]
    s_img = renderer.render(sV, sN, sF, camera)[0]

    t_img = alpha_blend(t_img)
    s_img = alpha_blend(s_img)

    return sum(losses) / len(losses), s_img, t_img

def normal_loss(target, source, alt=None, alt_n=None):
    batch = 10
    cameras = scene['view_mats']
    cameras_count = cameras.shape[0]
    losses = []

    tV, tN, tF = target[0], target[1].cuda(), target[2].cuda()
    sV, sN, sF = source[0], source[1].cuda(), source[2].cuda()

    if alt is not None:
        sF = alt

    if alt_n is not None:
        sN = alt_n

    assert cameras_count % batch == 0
    for i in range(0, cameras_count, batch):
        camera_batch = cameras[i:i + batch]
        t_nrms = renderer.render_normals(tV, tN, tF, camera_batch)
        s_nrms = renderer.render_normals(sV, sN, sF, camera_batch)

        loss = torch.mean(torch.abs(t_nrms - s_nrms))
        losses.append(loss.item())

    # Present view
    eye = torch.tensor([ 0, 0, -2.5 ], device='cuda')
    up = torch.tensor([0.0, -1.0, 0.0], device='cuda')
    center = torch.tensor([0.0, 0.0, 0.0], device='cuda')

    camera = lookat(eye, center, up).unsqueeze(0)

    t_nrm = renderer.render_normals(tV, tN, tF, camera)[0]
    s_nrm = renderer.render_normals(sV, sN, sF, camera)[0]

    return sum(losses) / len(losses), s_nrm, t_nrm
    # TODO: PSNR here (and L1)

def chamfer_loss(target, source):
    from tqdm import trange

    sV, _, _ = source[0], source[1].cuda(), source[2].cuda()
    tV, _, _ = target[0], target[1].cuda(), target[2].cuda()

    # First term
    sum_S1 = 0
    for i in trange(sV.shape[0]):
        # TODO: batch and use cdist
        sum_S1 += torch.min(torch.linalg.norm(sV[i] - tV, dim=1))
    sum_S1 /= sV.shape[0]

    # Second term
    sum_S2 = 0
    for i in trange(tV.shape[0]):
        sum_S2 += torch.min(torch.linalg.norm(tV[i] - sV, dim=1))
    sum_S2 /= tV.shape[0]

    return (sum_S1 + sum_S2).item()

# Writing to the database
import torchvision

def write(catag, data):
    global db

    # Open or create the database (json)
    if not os.path.exists(db):
        with open(db, 'w') as f:
            json.dump({}, f)

    with open(db, 'r') as f:
        db_json = json.load(f)

        # Save all images in disk and replace with paths
        count = data['count']

        image_paths = {}

        os.makedirs(os.path.join(directory, 'images'), exist_ok=True)
        for tag, img in data['images'].items():
            path = os.path.join(directory, 'images', '%s-%d-%s.png' % (catag, count, tag))
            img = img.permute(2, 0, 1)
            torchvision.utils.save_image(img.detach().cpu(), path)
            image_paths[tag] = path

        data['images'] = image_paths

        # Get the directory of the reference
        ref_dir = os.path.dirname(reference)
        ref_tag = os.path.basename(ref_dir)
        ref_entry = { catag: [ data ] }

        # Write the entry back
        if ref_tag in db_json:
            db_tmp = db_json[ref_tag]
            if catag in db_tmp:
                db_tmp[catag].append(ref_entry[catag][0])
                db_json[ref_tag] = db_tmp
            else:
                db_tmp[catag] = ref_entry[catag]
                db_json[ref_tag] = db_tmp
        else:
            db_json[ref_tag] = ref_entry

    # Write the database back
    with open(db, 'w') as f:
        json.dump(db_json, f, indent=4)

# Erase existing directory data from the database
if not os.path.exists(db):
    with open(db, 'w') as f:
        json.dump({}, f)
else:
    with open(db, 'r') as f:
        db_json = json.load(f)

        # get the directory of the reference
        ref_dir = os.path.dirname(reference)
        ref_tag = os.path.basename(ref_dir)

        if ref_tag in db_json:
            del db_json[ref_tag]

    # Write the database back
    with open(db, 'w') as f:
        json.dump(db_json, f, indent=4)

# Target mesh size
vertex_bytes = target[0].numel() * target[0].element_size()
index_bytes = target[2].numel() * target[2].element_size()
target_total = vertex_bytes + index_bytes

# Evaluating a single mesh
def evaluate(source):
    vertex_bytes = source.points.size * source.points.dtype.itemsize
    index_bytes = source.cells[0].data.size * source.cells[0].data.dtype.itemsize

    source.points = normalize(source.points)
    source = convert(source)

    fn = compute_face_normals(source[0], source[1])
    vn = compute_vertex_normals(source[0], source[1], fn)

    source = optext.geometry(source[0].cpu(), vn.cpu(), source[1].cpu())
    source_cas = optext.cached_grid(source, 32)
    source = source.torched()

    # Compute raw binary byte size
    vertex_bytes = source[0].numel() * source[0].element_size()
    index_bytes = source[2].numel() * source[2].element_size()
    total = vertex_bytes + index_bytes

    render, s_img, t_img = render_loss(target, source)
    normal, s_nrm, t_nrm = normal_loss(target, source)
    dpm, dnormal = sampled_measurements(target, target_cas, source, source_cas)
    chamfer      = chamfer_loss(target, source)
    cratio       = target_total / total

    return {
        'count': source[2].shape[0],
        'dpm': dpm,
        'dnormal': dnormal,
        'render': render,
        'normal': normal,
        'chamfer': chamfer,
        'size': total,
        'cratio': cratio,
        'images': {
            'render-source': s_img,
            'render-target': t_img,
            'normal-source': s_nrm,
            'normal-target': t_nrm
        }
    }

def evaluate_mesh(path):
    source = meshio.read(path)
    source.points = source.points.astype(np.float32)
    source.cells[0].data = source.cells[0].data.astype(np.int32)
    return evaluate(source)

# Recursively find all simple meshes and model configurations
for root, dirs, files in os.walk(directory):
    for file in files:
        path = os.path.join(root, file)
        if 'unpacked' in path:
            continue

        if file.endswith('.obj') and file.startswith('nvdiffmodeling'):
            print('Evaluating:', path)
            data = evaluate_mesh(path)
            write('Nvdiffmodeling', data)

        if not file.endswith('.pt'):
            continue

        # TODO: generate qslims... (then later run here...)
        print('Loading NGF representation:', file)

        data = torch.load(os.path.join(root, file))

        m = data['model']
        c = data['complexes']
        p = data['points']
        f = data['features']

        nsc = file.split('.')[0]

        m = data['model']
        c = data['complexes']
        p = data['points']
        f = data['features']
        l = data['kernel']
        ker = lerps[l]

        # Compute byte size of the representation
        feature_bytes = f.numel() * f.element_size()
        index_bytes = c.numel() * c.element_size()
        model_bytes = sum([ p.numel() * p.element_size() for p in m.parameters() ])
        vertex_bytes = p.numel() * p.element_size()
        total = feature_bytes + index_bytes + model_bytes + vertex_bytes

        rate = 16
        lerped_points, lerped_features = sample(c, p, f, rate, kernel=ker)

        vertices = m(points=lerped_points, features=lerped_features).detach()

        I = shorted_indices(vertices.cpu().numpy(), c, rate)
        I = torch.from_numpy(I).int()

        cmap = make_cmap(c, p.detach(), lerped_points.detach(), rate)
        remap = optext.generate_remapper(c.cpu(), cmap, lerped_points.shape[0], rate)
        F = remap.remap(I).cuda()

        fn = compute_face_normals(vertices, F)
        vn = compute_vertex_normals(vertices, F, fn)

        source = optext.geometry(vertices.cpu(), vn.cpu(), I)
        source_cas = optext.cached_grid(source, 32)
        source = source.torched()

        render, s_img, t_img = render_loss(target, source, alt=F, alt_n=vn)
        normal, s_nrm, t_nrm = normal_loss(target, source, alt=F, alt_n=vn)
        dpm, dnormal = sampled_measurements(target, target_cas, source, source_cas)
        chamfer      = chamfer_loss(target, source)
        cratio       = target_total / total

        data = {
            'count': c.shape[0],
            'dpm': dpm,
            'dnormal': dnormal,
            'render': render,
            'normal': normal,
            'chamfer': chamfer,
            'size': total,
            'cratio': cratio,
            'images': {
                'render-source': s_img,
                'render-target': t_img,
                'normal-source': s_nrm,
                'normal-target': t_nrm
            }
        }

        write('NGF', data)

        # Find qslim for this mesh at comparable compression ratio
        binary = os.path.join(os.path.dirname(__file__), 'build', 'simplify')
        base = os.path.basename(reference)
        base = os.path.splitext(base)[0]
        os.makedirs(os.path.join(directory, 'comparisons'), exist_ok=True)
        smashed = os.path.join(directory, 'comparisons', base + '_smashed.obj')

        def qslim(rate):
            cmd = subprocess.list2cmdline([ binary, reference, smashed, str(rate) ])
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Compute raw binary byte size
            qslimmed = meshio.read(smashed)
            qslimmed.points = qslimmed.points.astype(np.float32)
            qslimmed.cells[0].data = qslimmed.cells[0].data.astype(np.int32)

            vertex_bytes = qslimmed.points.size * qslimmed.points.dtype.itemsize
            index_bytes = qslimmed.cells[0].data.size * qslimmed.cells[0].data.dtype.itemsize
            total = vertex_bytes + index_bytes

            # print('  > QSlim: %s [%f] at %d vertices' % (smashed, target_total / total, qslimmed.points.shape[0]))
            return target_total / total

        small, large = 0.001, 0.95
        for i in range(15):
            mid = (small + large) / 2
            qslim_cratio = qslim(mid)
            if qslim_cratio < cratio:
                large = mid
            else:
                small = mid

        # Process this mesh as well
        write('QSlim', evaluate_mesh(smashed))
        save = os.path.join(directory, 'comparisons', 'qslim-%s-%d.obj' % (nsc, c.shape[0]))
        shutil.copy(smashed, save)
