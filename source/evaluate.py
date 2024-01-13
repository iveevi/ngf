import os
import torch
import optext
import argparse
import numpy as np

from ngf import load_ngf
from mesh import Mesh, mesh_from, load_mesh
from util import make_cmap, arrange_views, lookat
from render import Renderer

def mesh_size(V, F):
    vertices_size = V.numel() * V.element_size()
    faces_size = F.numel() * F.element_size()
    return vertices_size + faces_size

# TODO: also util function...
def construct_renderer():
    import imageio.v2 as imageio

    path = os.path.join(os.path.dirname(__file__), '../images/environment.hdr')
    path = os.path.abspath(path)
    environment = imageio.imread(path, format='HDR')
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
        self.preview = True

    # TODO: util function
    def postprocess(self, f):
        f = torch.log(torch.clamp(f, min=0, max=65535) + 1)
        return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4) * 1.055 - 0.055, 12.92 * f)

    def get_view(self, tag):
        predef = {
                'xyz': torch.tensor([ 2, 0, 1 ], device='cuda').float(),
                'einstein': torch.tensor([ 0, 0, 3.5 ], device='cuda').float(),
                'skull': torch.tensor([ -0.5, 0, 2.5 ], device='cuda').float(),
        }

        eye    = torch.tensor([ 0, 0, 3 ], device='cuda').float()
        up     = torch.tensor([ 0, 1, 0 ], device='cuda').float()
        center = torch.tensor([ 0, 0, 0 ], device='cuda').float()

        if tag in predef:
            eye = predef[tag]

        look = center - eye
        if torch.dot(look, up).abs().item() > 1.0 - 1e-6:
            up = torch.tensor([1, 0, 0], device='cuda').float()
        if torch.dot(look, up).abs().item() > 1.0 - 1e-6:
            up = torch.tensor([0, 0, 1], device='cuda').float()

        right = torch.cross(look, up)
        right /= right.norm()

        up = torch.cross(look, right)
        up /= up.norm()

        look /= look.norm()

        return torch.tensor([
            [ right[0], up[0], look[0], eye[0] ],
            [ right[1], up[1], look[1], eye[1] ],
            [ right[2], up[2], look[2], eye[2] ],
            [ 0, 0, 0, 1 ]
        ], dtype=torch.float32, device='cuda').inverse()

    def eval_render(self, mesh, tag):
        batch = 10
        views = torch.split(self.views, batch)

        errors = []
        for batch_views in views:
            ref_imgs = self.renderer.render_spherical_harmonics(self.reference.vertices, self.reference.normals, self.reference.faces, batch_views)
            mesh_imgs = self.renderer.render_spherical_harmonics(mesh.vertices, mesh.normals, mesh.faces, batch_views)

            error = torch.mean(torch.abs(ref_imgs - mesh_imgs))
            errors.append(error.item())

        camera = self.get_view(tag).unsqueeze(0)
        ref_img = self.renderer.render_spherical_harmonics(self.reference.vertices, self.reference.normals, self.reference.faces, camera)[0]
        mesh_img = self.renderer.render_spherical_harmonics(mesh.vertices, mesh.normals, mesh.faces, camera)[0]

        return { 'error': np.mean(errors), 'ref': ref_img, 'mesh': mesh_img }

    def eval_normals(self, mesh, tag):
        batch = 10
        views = torch.split(self.views, batch)

        errors = []
        for batch_views in views:
            ref_imgs = self.renderer.render_normals(self.reference.vertices, self.reference.normals, self.reference.faces, batch_views)
            mesh_imgs = self.renderer.render_normals(mesh.vertices, mesh.normals, mesh.faces, batch_views)

            error = torch.mean(torch.abs(ref_imgs - mesh_imgs))
            errors.append(error.item())

        camera = self.get_view(tag).unsqueeze(0)
        ref_img = self.renderer.render_normals(self.reference.vertices, self.reference.normals, self.reference.faces, camera)[0]
        mesh_img = self.renderer.render_normals(mesh.vertices, mesh.normals, mesh.faces, camera)[0]

        # if self.preview:
        #     self.preview = False
        #     import matplotlib.pyplot as plt
        #     plt.imshow(ref_img.cpu().numpy())
        #     plt.show()

        return { 'error': np.mean(errors), 'ref': ref_img, 'mesh': mesh_img }

    def eval_chamfer(self, mesh):
        from kaolin.metrics.pointcloud import chamfer_distance
        error = chamfer_distance(mesh.vertices.unsqueeze(0), self.reference.vertices.unsqueeze(0))
        return { 'error': error.item() }

    def eval_metrics(self, mesh, tag=None):
        render = self.eval_render(mesh, tag)
        normals = self.eval_normals(mesh, tag)
        chamfer = self.eval_chamfer(mesh)
        return {
            'render': render['error'],
            'normal': normals['error'],
            'chamfer': chamfer['error'],
            'images': {
                'render:ref': render['ref'],
                'render:mesh': render['mesh'],
                'normal:ref': normals['ref'],
                'normal:mesh': normals['mesh']
            }
        }

def ngf_size(ngf):
    size = sum([ p.numel() * p.element_size() for p in ngf.parameters()])
    size += ngf.complexes.numel() * ngf.complexes.element_size()
    return size

def scene_evaluations(reference, directory, alt=None):
    import re
    import sys
    import json
    import subprocess

    name = os.path.basename(directory)
    print('Evaluating scene', name)

    ref, _ = load_mesh(reference)
    ref_size = mesh_size(ref.vertices, ref.faces)
    key = os.path.basename(directory)
    pattern = re.compile('lod[1-4].pt$')

    evaluator = Evaluator(ref)

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

        return size

    def qslim_search(size):
        from tqdm import trange

        small, large = 0.001, 0.95
        for _ in trange(15, ncols=100, desc='QSlim'):
            mid = (small + large) / 2
            qsize = qslim_exec(mid)
            if qsize > size:
                large = mid
            else:
                small = mid

        qslim_exec((small + large) / 2)

    evaluations = { 'reference' : {
            'size': mesh_size(ref.vertices, ref.faces),
            'count': ref.faces.shape[0]
        }
    }

    sizes = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            if not pattern.match(file):
                continue

            print('Loading NGF from', file)

            ngf = torch.load(path)
            ngf = load_ngf(ngf)
            size = ngf_size(ngf)

            uvs = ngf.sample_uniform(16)
            V = ngf.eval(*uvs).detach()

            base = ngf.base(16).detach()
            cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, 16)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], 16)
            indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], 16)
            F = remap.remap_device(indices)

            ngf_mesh = mesh_from(V, F)

            metrics = evaluator.eval_metrics(ngf_mesh, name)

            metrics['count'] = ngf.complexes.shape[0]
            # count = ngf.complexes.shape[0]
            metrics['size'] = size
            metrics['cratio'] = ref_size/size

            # evaluations.setdefault('Ours', {})
            # evaluations['Ours'][count] = metrics
            evaluations.setdefault('Ours', []).append(metrics)
            sizes.append(size)


    # QSlim and nvdiffmodeling at various sizes
    for size_kb in sizes:
        # QSlim
        qslim_search(1024 * size_kb)

        smashed, _ = load_mesh(qslim_result)

        metrics = evaluator.eval_metrics(smashed, name)
        metrics['count'] = smashed.faces.shape[0]
        metrics['size'] = mesh_size(smashed.vertices, smashed.faces)
        metrics['cratio'] = ref_size/metrics['size']

        evaluations.setdefault('QSlim', []).append(metrics)

        # nvdiffmodeling
        data = {
                'base_mesh': qslim_result,
                'ref_mesh': alt if alt else reference,
                'camera_eye': [ 2.5, 0.0, -2.5 ],
                'camera_up': [ 0.0, 1.0, 0.0 ],
                'random_textures': True,
                'iter': 5000,
                'save_interval': 250,
                'train_res': 512,
                'batch': 8,
                'learning_rate': 0.001,
                'min_roughness' : 0.25,
                'out_dir' : os.path.join('evals', 'nvdiffmodeling', name)
        }

        print('running with config', data)

        PYTHON = sys.executable
        SCRIPT = os.environ['HOME'] + '/sources/nvdiffmodeling/train.py'

        os.makedirs('evals/nvdiffmodeling', exist_ok=True)
        config = os.path.join('evals', 'nvdiffmodeling', name + '.json')
        with open(config, 'w') as f:
            json.dump(data, f, indent=4)

        cmd = '{} {} --config {}'.format(PYTHON, SCRIPT, config)
        print('cmd', cmd)

        subprocess.run(cmd.split())

        nvdiff, _ = load_mesh(os.path.join('evals', 'nvdiffmodeling', name, 'mesh', 'mesh.obj'))

        metrics = evaluator.eval_metrics(nvdiff)
        metrics['count'] = nvdiff.faces.shape[0]
        metrics['size'] = mesh_size(nvdiff.vertices, nvdiff.faces)

        evaluations.setdefault('nvdiffmodeling', []).append(metrics)

    os.makedirs('evals', exist_ok=True)
    torch.save(evaluations, os.path.join('evals', name + '.pt'))

def ngf_mesh(ngf, rate=16) -> Mesh:
    with torch.no_grad():
        uvs = ngf.sample_uniform(rate)
        V = ngf.eval(*uvs)
        base = ngf.base(rate)

    cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
    remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
    indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
    F = remap.remap_device(indices)

    return mesh_from(V, F)

def tessellation_evaluation(prefix):
    print('tess prefix:', prefix)

    rdir = os.path.abspath(os.path.join(__file__, os.pardir))
    rdir = os.path.abspath(os.path.join(rdir, os.pardir))
    base = os.path.abspath(os.path.join(rdir, 'results'))

    def eval_tessellations(evaluator, ngf):
        rate_metrics = {}
        for rate in [ 2, 4, 8, 12, 16, 32 ]:
            mesh = ngf_mesh(ngf, rate)
            metrics = evaluator.eval_metrics(mesh)
            rate_metrics[rate] = {
                'render': metrics['render'],
                'normal': metrics['normal'],
                'chamfer': metrics['chamfer']
            }

        return rate_metrics

    data = {}
    for root, _, files in os.walk(base):
        for file in files:
            if prefix in file and file.endswith('pt'):
                file = os.path.join(root, file)
                scene = os.path.basename(root)
                reference = os.path.join(rdir, 'meshes', scene, 'target.obj')
                reference, _ = load_mesh(reference)
                evaluator = Evaluator(reference)

                ngf = torch.load(file)
                ngf = load_ngf(ngf)

                rm = eval_tessellations(evaluator, ngf)
                data[scene] = rm

    print(data)

    import json
    with open('tessellation.json', 'w') as f:
        json.dump(data, f)

def feature_evaluation(prefix):
    print('feature prefix:', prefix)

    rdir = os.path.abspath(os.path.join(__file__, os.pardir))
    rdir = os.path.abspath(os.path.join(rdir, os.pardir))
    base = os.path.abspath(os.path.join(rdir, 'results'))

    # Evaluators
    setup = {}

    for root, _, files in os.walk(base):
        for file in files:
            if prefix in file and file.endswith('pt'):
                file = os.path.join(root, file)
                scene = os.path.basename(root)

                print('file is', file)
                setup.setdefault(scene, {})
                if 'evaluator' not in setup[scene]:
                    reference = os.path.join(rdir, 'meshes', scene, 'target.obj')
                    reference, _ = load_mesh(reference)
                    evaluator = Evaluator(reference)
                    setup[scene]['evaluator'] = evaluator

                ngf = torch.load(file)
                ngf = load_ngf(ngf)

                features = ngf.features.shape[1]
                setup[scene][features] = ngf

    # Gather metrics
    data = {}
    for scene, components in setup.items():
        print('Evaluating ', scene)

        evaluator = components['evaluator']
        for k in components:
            if type(k) != int:
                continue

            ngf = components[k]

            uvs = ngf.sample_uniform(16)
            V = ngf.eval(*uvs).detach()

            base = ngf.base(16).detach()
            cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, 16)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], 16)
            indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], 16)
            F = remap.remap_device(indices)

            ngf_mesh = mesh_from(V, F)

            size = sum([ p.numel() * p.element_size() for p in ngf.parameters()])
            size += ngf.complexes.numel() * ngf.complexes.element_size()

            metrics = evaluator.eval_metrics(ngf_mesh)
            data.setdefault(scene, {})[k] = {
                'render': metrics['render'],
                'normal': metrics['normal'],
                'chamfer': metrics['chamfer'],
                'size': size
            }

    with open('features.json', 'w') as f:
        import json
        json.dump(data, f)

def multichart_evaluations():
    import re
    import sys

    sys.path.append('experiments/mcgim')

    # NOTE: Using nefertiti as the model for this ablation

    ngf_directory = 'results/nefertiti'
    mcgim_directory = 'experiments/mcgim/results/nefertiti'
    neu_mcgim_directory = mcgim_directory

    assert os.path.exists(ngf_directory)
    assert os.path.exists(mcgim_directory)
    assert os.path.exists(neu_mcgim_directory)

    # Data to plot (Chamfer only)
    data = { 'ngf': [], 'mcgim': [], 'neural-mcgim': [] }

    # Configure the evaluator
    reference, _ = load_mesh('meshes/nefertiti/target.obj')
    evaluator = Evaluator(reference)

    # Find all ngf models
    def ngf_to_mesh(ngf):
        sample = ngf.sample_uniform(16)
        V = ngf.eval_oo(*sample).detach()

        base = ngf.base(16).detach()
        cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, 16)
        remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], 16)
        indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], 16)
        F = remap.remap_device(indices)

        return mesh_from(V, F)

    pattern = re.compile('^lod[1-4].pt$')
    for root, _, files in os.walk(ngf_directory):
        for file in files:
            if pattern.match(file):
                print('Found ngf file', file)

                file = os.path.join(root, file)
                size = os.path.getsize(file)
                ngf = torch.load(file)
                ngf = load_ngf(ngf)

                print('  > size', size // 1024, 'KB')

                mesh = ngf_to_mesh(ngf)
                metrics = evaluator.eval_metrics(mesh)
                del metrics['images']
                print('  > metrics', metrics)

                data['ngf'].append((size, metrics['chamfer']))

    # Find all mcgim models
    def mcgim_to_mesh(mcgim):
        total_vertices = []
        total_indices = []

        sampling = mcgim.shape[1]
        for k, gim in enumerate(mcgim):
            gim = gim.reshape(-1, 3)
            gim_numpy = gim.cpu().numpy()

            indices = []
            for i in range(sampling - 1):
                for j in range(sampling - 1):
                    a = i * sampling + j
                    c = (i + 1) * sampling + j
                    b, d = a + 1, c + 1
                    indices.append([a, b, c])
                    indices.append([b, d, c])

                    vs = gim_numpy[[a, b, c, d]]
                    d0 = np.linalg.norm(vs[0] - vs[3])
                    d1 = np.linalg.norm(vs[1] - vs[2])

                    if d0 < d1:
                        indices.append([a, b, d])
                        indices.append([a, d, c])
                    else:
                        indices.append([a, b, c])
                        indices.append([b, d, c])

            indices = np.array(indices)
            indices += k * sampling ** 2

            total_indices.append(torch.from_numpy(indices).int().cuda())
            total_vertices.append(gim)

        total_vertices = torch.cat(total_vertices)
        total_indices = torch.cat(total_indices)

        # print('total_vertices', total_vertices.shape)
        # print('total_indices', total_indices.shape)
        #
        # import polyscope as ps
        #
        # ps.init()
        # ps.register_surface_mesh('m', total_vertices.cpu().numpy(), total_indices.cpu().numpy())
        # ps.show()

        return mesh_from(total_vertices, total_indices)

    pattern = re.compile(r'^mcgim-\d+-\d+.pt$')
    for root, _, files in os.walk(mcgim_directory):
        for file in files:
            if pattern.match(file):
                print('Found mcgim file', file)

                file = os.path.join(root, file)
                size = os.path.getsize(file)
                mcgim = torch.load(file)

                print('  > mcgim shape', mcgim.shape)
                print('  > size', size // 1024, 'KB')

                mesh = mcgim_to_mesh(mcgim)
                metrics = evaluator.eval_metrics(mesh)
                del metrics['images']
                print('  > metrics', metrics)

                data['mcgim'].append((size, metrics['chamfer']))

    # Find all neural mcgim models
    def neural_mcgim_to_mesh(mcgim, size):
        print('mcgim shape', mcgim.shape, size)

        total_vertices = []
        total_indices = []

        N, M, sampling = size
        for k in range(N * M):
            x, y = k % N, k // N

            gim = mcgim[x * sampling : (x + 1) * sampling, y * sampling : (y + 1) * sampling]
            gim = gim.reshape(-1, 3)
            gim_numpy = gim.cpu().numpy()

            indices = []
            for i in range(sampling - 1):
                for j in range(sampling - 1):
                    a = i * sampling + j
                    c = (i + 1) * sampling + j
                    b, d = a + 1, c + 1
                    indices.append([a, b, c])
                    indices.append([b, d, c])

                    vs = gim_numpy[[a, b, c, d]]
                    d0 = np.linalg.norm(vs[0] - vs[3])
                    d1 = np.linalg.norm(vs[1] - vs[2])

                    if d0 < d1:
                        indices.append([a, b, d])
                        indices.append([a, d, c])
                    else:
                        indices.append([a, b, c])
                        indices.append([b, d, c])

            indices = np.array(indices)
            indices += k * sampling ** 2

            total_indices.append(torch.from_numpy(indices).int().cuda())
            total_vertices.append(gim)

        total_vertices = torch.cat(total_vertices)
        total_indices = torch.cat(total_indices)

        return mesh_from(total_vertices, total_indices)

    pattern = re.compile(r'^neural-mcgim-\d+-\d+.pt$')
    for root, _, files in os.walk(mcgim_directory):
        for file in files:
            if pattern.match(file):
                print('Found neural mcgim file', file)

                file = os.path.join(root, file)
                size = os.path.getsize(file)
                packet = torch.load(file)

                model = packet['model']
                mcgim = model.evaluate(**packet)

                print('  > mcgim shape', mcgim.shape)
                print('  > size', size // 1024, 'KB')

                mesh = neural_mcgim_to_mesh(mcgim, packet['sampling'])
                metrics = evaluator.eval_metrics(mesh)
                del metrics['images']
                print('  > metrics', metrics)

                data['neural-mcgim'].append((size, metrics['chamfer']))

    print('data', data)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.dpi'] = 300

    sns.set_theme()
    for key in data:
        data[key].sort(key=lambda v: v[0])
        sizes = [ v[0] // 1024 for v in data[key] ]
        error = [ v[1] for v in data[key] ]
        plt.plot(sizes, error, label=key)

    plt.xlabel('Size (KB)')
    plt.ylabel('Chamfer distance')
    plt.yscale('log')
    plt.legend()

    plt.savefig('gims.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=str, help='path to reference mesh')
    parser.add_argument('--alt', type=str, help='path to alternative reference mesh')
    parser.add_argument('--results', type=str, help='path to results directory')
    parser.add_argument('--tess-prefix', type=str, help='prefix for tessellation evaluations')
    parser.add_argument('--feat-prefix', type=str, help='prefix for feature size evaluations')
    parser.add_argument('--multichart', action='store_true', help='perform multichart geometry images ablations')
    args = parser.parse_args()

    if args.results:
        scene_evaluations(args.reference, args.results, args.alt)
    elif args.tess_prefix:
        tessellation_evaluation(args.tess_prefix)
    elif args.feat_prefix:
        feature_evaluation(args.feat_prefix)
    elif args.multichart:
        multichart_evaluations()
    else:
        raise NotImplementedError
