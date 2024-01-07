import os
import torch
import optext
import argparse
import imageio
import numpy as np

from ngf import load_ngf
from mesh import Mesh, mesh_from, load_mesh
from util import make_cmap, lookat
from render import Renderer

def mesh_size(V, F):
    vertices_size = V.numel() * V.element_size()
    faces_size = F.numel() * F.element_size()
    return vertices_size + faces_size

# TODO: util function
def arrange_views(mesh: Mesh, cameras: int):
    seeds = list(torch.randint(0, mesh.faces.shape[0], (cameras,)).numpy())
    clusters = optext.cluster_geometry(mesh.optg, seeds, 3, 'uniform')

    cluster_centroids = []
    cluster_normals = []

    for cluster in clusters:
        faces = mesh.faces[cluster]

        v0 = mesh.vertices[faces[:, 0]]
        v1 = mesh.vertices[faces[:, 1]]
        v2 = mesh.vertices[faces[:, 2]]
        centroids = (v0 + v1 + v2) / 3.0
        centroids = centroids.mean(dim=0)


        normals = torch.cross(v1 - v0, v2 - v0)
        normals = normals.mean(dim=0)
        normals = normals / torch.norm(normals)

        cluster_centroids.append(centroids)
        cluster_normals.append(normals)

    cluster_centroids = torch.stack(cluster_centroids, dim=0)
    cluster_normals = torch.stack(cluster_normals, dim=0)

    # Generate camera views
    canonical_up = torch.tensor([0.0, 1.0, 0.0], device='cuda')
    cluster_eyes = cluster_centroids + cluster_normals * 1.0
    cluster_ups = torch.stack(len(clusters) * [ canonical_up ], dim=0)
    cluster_rights = torch.cross(cluster_normals, cluster_ups)
    cluster_ups = torch.cross(cluster_rights, cluster_normals)

    all_views = [ lookat(eye, view_point, up) for eye, view_point, up in zip(cluster_eyes, cluster_centroids, cluster_ups) ]
    all_views = torch.stack(all_views, dim=0)

    return all_views

# TODO: also util function...
def construct_renderer():
    path = os.path.join(os.path.dirname(__file__), '../images/environment.hdr')
    environment = imageio.imread(path, format='HDR-FI')
    environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
    alpha       = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
    environment = torch.cat((environment, alpha), dim=-1)

    return Renderer(width=1280, height=720, fov=45.0, near=0.1, far=1000.0, envmap=environment)

class Evaluator:
    CAMERAS = 100

    def __init__(self, reference):
        self.reference = reference
        self.ref_size = mesh_size(reference.vertices, reference.faces)
        print('ref_size =', self.ref_size)
        self.views = arrange_views(reference, Evaluator.CAMERAS)
        print('views:', self.views.shape)
        self.renderer = construct_renderer()

    # TODO: util function
    def postprocess(self, f):
        f = torch.log(torch.clamp(f, min=0, max=65535) + 1)
        return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4) * 1.055 - 0.055, 12.92 * f)

    def eval_render(self, mesh):
        batch = 10
        views = torch.split(self.views, batch)

        errors = []
        for batch_views in views:
            ref_imgs = self.renderer.render_spherical_harmonics(self.reference.vertices, self.reference.normals, self.reference.faces, batch_views)
            mesh_imgs = self.renderer.render_spherical_harmonics(mesh.vertices, mesh.normals, mesh.faces, batch_views)

            error = torch.mean(torch.abs(ref_imgs - mesh_imgs))
            errors.append(error.item())

        # Present view
        # TODO: should be configurable
        eye    = torch.tensor([ 0, 0, -2.5 ], device='cuda')
        up     = torch.tensor([0.0, -1.0, 0.0], device='cuda')
        center = torch.tensor([0.0, 0.0, 0.0], device='cuda')

        camera = lookat(eye, center, up).unsqueeze(0)

        ref_img = self.renderer.render_spherical_harmonics(self.reference.vertices, self.reference.normals, self.reference.faces, camera)[0]
        mesh_img = self.renderer.render_spherical_harmonics(mesh.vertices, mesh.normals, mesh.faces, camera)[0]

        return { 'error': np.mean(errors), 'ref': ref_img, 'mesh': mesh_img }

    def eval_normals(self, mesh):
        batch = 10
        views = torch.split(self.views, batch)

        errors = []
        for batch_views in views:
            ref_imgs = self.renderer.render_normals(self.reference.vertices, self.reference.normals, self.reference.faces, batch_views)
            mesh_imgs = self.renderer.render_normals(mesh.vertices, mesh.normals, mesh.faces, batch_views)

            error = torch.mean(torch.abs(ref_imgs - mesh_imgs))
            errors.append(error.item())

        # Present view
        # TODO: should be configurable
        eye    = torch.tensor([ 0, 0, -2.5 ], device='cuda')
        up     = torch.tensor([0.0, -1.0, 0.0], device='cuda')
        center = torch.tensor([0.0, 0.0, 0.0], device='cuda')

        camera = lookat(eye, center, up).unsqueeze(0)

        ref_img = self.renderer.render_normals(self.reference.vertices, self.reference.normals, self.reference.faces, camera)[0]
        mesh_img = self.renderer.render_normals(mesh.vertices, mesh.normals, mesh.faces, camera)[0]

        return { 'error': np.mean(errors), 'ref': ref_img, 'mesh': mesh_img }

    def eval_chamfer(self, mesh):
        from kaolin.metrics.pointcloud import chamfer_distance
        error = chamfer_distance(mesh.vertices.unsqueeze(0), self.reference.vertices.unsqueeze(0))
        return { 'error': error.item() }

    def eval_metrics(self, mesh):
        render = self.eval_render(mesh)
        normals = self.eval_normals(mesh)
        chamfer = self.eval_chamfer(mesh)
        return {
            'render': render['error'],
            'normal': normals['error'],
            'chamfer': chamfer['error'],
            'images': {
                'render:ref': render['ref'],
                'render:mesh': render['mesh'],
                'normal:ref': normals['ref'],
                'normal:mesh': normals['mesh'],
            }
        }

def ngf_size(ngf):
    size = sum([ p.numel() * p.element_size() for p in ngf.parameters()])
    size += ngf.complexes.numel() * ngf.complexes.element_size()
    return size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=str, help='path to reference')
    parser.add_argument('ngf', type=str, help='path to ngf')
    args = parser.parse_args()

    reference, _ = load_mesh(args.reference)

    evaluator = Evaluator(reference)

    ngf = torch.load(args.ngf)
    ngf = load_ngf(ngf)
    print(ngf)

    RATE = 16

    uvs = ngf.sample_uniform(RATE)
    V = ngf.eval(*uvs)

    base = ngf.base(RATE).detach()
    cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, RATE)
    remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], RATE)
    indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], RATE)
    F = remap.remap_device(indices)

    ngf_mesh = mesh_from(V, F)

    # print(ngf_mesh)

    ref_size = mesh_size(reference.vertices, reference.faces)

    metrics = evaluator.eval_metrics(ngf_mesh)
    metrics['size'] = ngf_size(ngf)
    metrics['cratio'] = ref_size/metrics['size']
    metrics['size'] //= 1024
    del metrics['images']
    print(metrics)

    # print('Target size %d kB' % (metrics['size'] // 1024))

    qslim = os.path.join(os.path.dirname(__file__), os.path.pardir, 'build', 'simplify')
    assert os.path.exists(qslim), 'Could not find simplification program %s' % qslim

    os.makedirs('results/generated', exist_ok=True)
    qslim_result = 'results/generated/smashed.obj'

    def qslim_exec(rate):
        import subprocess

        cmd = subprocess.list2cmdline([ qslim, args.reference, qslim_result, str(rate) ])
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        qslimmed, _ = load_mesh(qslim_result)
        size = mesh_size(qslimmed.vertices, qslimmed.faces)
        print('  > QSlim [%f] has size %d kB' % (rate, size // 1024))

        return size // 1024

    # small, large = 0.001, 0.95
    # for i in range(15):
    #     mid = (small + large) / 2
    #     size = qslim_exec(mid)
    #     if size > metrics['size']:
    #         large = mid
    #     else:
    #         small = mid
    #
    # qslim_exec((small + large) / 2)

    # # Instant NGP
    import pyngp as ngp
    import time

    from tqdm import tqdm

    STEPS = 100

    def new_testbed():
        testbed = ngp.Testbed()
        testbed.root_dir = 'results/generated'

        testbed.load_file(args.reference)

        testbed.background_color = [0.580, 0.713, 0.882, 1.000]
        testbed.exposure = 1.000
        testbed.sun_dir = [0.541,-0.839,-0.042]

        testbed.sdf.brdf.metallic = 0.000
        testbed.sdf.brdf.subsurface = 0.000
        testbed.sdf.brdf.specular = 1.000
        testbed.sdf.brdf.roughness = 0.300
        testbed.sdf.brdf.sheen = 0.000
        testbed.sdf.brdf.clearcoat = 0.000
        testbed.sdf.brdf.clearcoat_gloss = 0.000
        testbed.sdf.brdf.basecolor = [0.800,0.800,0.800]

        testbed.autofocus_target = [0.500,0.500,0.500]
        testbed.autofocus = False

        testbed.sdf.analytic_normals = False
        testbed.sdf.use_triangle_octree = False

        col = list(testbed.background_color)
        testbed.sdf.brdf.ambientcolor = np.multiply(col,col)[0:3]
        # testbed.sdf.shadow_sharpness = 16 if softshadow else 2048
        testbed.sdf.shadow_sharpness = 2048
        testbed.scale = testbed.scale * 1.13

        return testbed

    # TODO: write manually here...
    import json

    config = json.load(open('/home/venki/sources/instant-ngp/configs/sdf/base.json', 'r'))

    def run_testbed(log_size, features=None):
        config['encoding']['log2_hashmap_size'] = log_size
        if features is not None:
            config['encoding']['n_features_per_level'] = features
        else:
            config['encoding']['n_features_per_level'] = 2

        with open('results/generated/ingp/config.json', 'w') as f:
            json.dump(config, f)

        old_training_step = 0
        tqdm_last_update = 0

        testbed = new_testbed()
        testbed.reload_network_from_file('results/generated/ingp/config.json')

        with tqdm(desc="Training", total=STEPS, unit="steps") as t:
            while testbed.frame():
                # What will happen when training is done?
                if testbed.training_step >= STEPS:
                    break

                # Update progress bar
                if testbed.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()

                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now

        os.makedirs('results/generated/ingp', exist_ok=True)
        testbed.save_snapshot('results/generated/ingp/proxy.ingp', False)

        size = os.path.getsize('results/generated/ingp/proxy.ingp')
        # print('Size (kB)', size // 1024)
        return testbed, size

    # configs = [ (10, 2), (10, 4), (11, 1), (11, 2), (11, 4), (12, None), (13, None) ]
    #
    # closest = None
    # diff    = 1e20
    #
    # for c in configs:
    #     _, size = run_testbed(*c)
    #     if abs(metrics['size'] - size) < diff:
    #         diff = metrics['size'] - size
    #         closest = c
    #
    # print('Closest config is', closest)

    # STEPS = 10_000
    #
    # testbed, _ = run_testbed(13, 2)
    # testbed.init_window(1920, 1080)
    # testbed.shall_train = False
    # while testbed.frame():
    #     continue

    # Load both
    assert os.path.exists(qslim_result)
    assert os.path.exists('results/generated/ingp.obj')

    smashed, _ = load_mesh(qslim_result)
    ingp,    _ = load_mesh('results/generated/ingp.obj')

    metrics = evaluator.eval_metrics(smashed)
    metrics['size'] = mesh_size(smashed.vertices, smashed.faces)
    metrics['cratio'] = ref_size/metrics['size']
    metrics['size'] //= 1024
    del metrics['images']
    print(metrics)

    metrics = evaluator.eval_metrics(ingp)
    metrics['size'] = os.path.getsize('results/generated/ingp/proxy.ingp')
    metrics['cratio'] = ref_size/metrics['size']
    metrics['size'] //= 1024
    del metrics['images']
    print(metrics)

    import polyscope as ps

    ps.init()
    ps.register_surface_mesh('ngf', ngf_mesh.vertices.detach().cpu().numpy(), ngf_mesh.faces.cpu().numpy())
    ps.register_surface_mesh('qslim', smashed.vertices.cpu().numpy(), smashed.faces.cpu().numpy())
    ps.register_surface_mesh('ingp', ingp.vertices.cpu().numpy(), ingp.faces.cpu().numpy())
    ps.show()
