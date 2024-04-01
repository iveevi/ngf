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
                'xyz'       : torch.tensor([ 2, 0, 1 ], device='cuda').float(),
                'einstein'  : torch.tensor([ 0, 0, 3.5 ], device='cuda').float(),
                'skull'     : torch.tensor([ 0.5, 0, 2.5 ], device='cuda').float(),
                # 'skull'     : torch.tensor([ -0.5, 0, 2.5 ], device='cuda').float(),
                # 'skull'     : torch.tensor([ 2, 0, 2.5 ], device='cuda').float(),
                'wreck'     : torch.tensor([ 0, 0, 2 ], device='cuda').float(),
                'armadillo' : torch.tensor([ -1.8, 0, -2.8 ], device='cuda').float(),
                'nefertiti' : torch.tensor([ 0, 0, -3.5 ], device='cuda').float(),
                'lucy'      : torch.tensor([ 0, 0, -4.0 ], device='cuda').float(),
                'dragon'    : torch.tensor([ 0, 0, 3.6 ], device='cuda').float(),
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

    def eval_normals(self, mesh, tag, invert=False):
        batch = 10
        views = torch.split(self.views, batch)

        errors = []
        for batch_views in views:
            ref_imgs = self.renderer.render_normals(self.reference.vertices, self.reference.normals, self.reference.faces, batch_views)
            mesh_imgs = self.renderer.render_normals(mesh.vertices, mesh.normals, mesh.faces, batch_views, invert=invert)

            error = torch.mean(torch.abs(ref_imgs - mesh_imgs))
            errors.append(error.item())

        camera = self.get_view(tag).unsqueeze(0)
        ref_img = self.renderer.render_normals(self.reference.vertices, self.reference.normals, self.reference.faces, camera)[0]
        mesh_img = self.renderer.render_normals(mesh.vertices, mesh.normals, mesh.faces, camera, invert=invert)[0]

        return { 'error': np.mean(errors), 'ref': ref_img, 'mesh': mesh_img }

    def eval_chamfer(self, mesh):
        from kaolin.metrics.pointcloud import chamfer_distance
        error = chamfer_distance(mesh.vertices.unsqueeze(0), self.reference.vertices.unsqueeze(0))
        return { 'error': error.item() }

    def eval_metrics(self, mesh, tag=None, invert=False):
        render = self.eval_render(mesh, tag)
        normals = self.eval_normals(mesh, tag, invert)
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

    def render_everything(self, mesh, tag=None, invert=False):
        camera = self.get_view(tag).unsqueeze(0)

        nrm_ref_img = self.renderer.render_normals(self.reference.vertices, self.reference.normals, self.reference.faces, camera)[0]
        nrm_mesh_img = self.renderer.render_normals(mesh.vertices, mesh.normals, mesh.faces, camera, invert=invert)[0]

        sph_ref_img = self.renderer.render_spherical_harmonics(self.reference.vertices, self.reference.normals, self.reference.faces, camera)[0]
        sph_mesh_img = self.renderer.render_spherical_harmonics(mesh.vertices, mesh.normals, mesh.faces, camera)[0]

        # if self.preview:
        #     self.preview = False
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme()
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(nrm_ref_img.cpu().numpy())
        axs[0].axis('off')
        axs[1].imshow(sph_ref_img.cpu().numpy())
        axs[1].axis('off')
        plt.show()

        return {
            'sph:ref'  :  sph_ref_img,
            'sph:mesh' :  sph_mesh_img,
            'nrm:ref'  :  nrm_ref_img,
            'nrm:mesh' :  nrm_mesh_img,
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
    # pattern = re.compile('lod[1-4].pt$')
    pattern = re.compile('lod1.pt$')

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
            metrics['size'] = size
            metrics['cratio'] = ref_size/size

            evaluations.setdefault('Ours', []).append(metrics)
            sizes.append(size)

    # QSlim and nvdiffmodeling at various sizes
    for size in sizes:
        # QSlim
        qslim_search(size)

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

        metrics = evaluator.eval_metrics(nvdiff, name)
        metrics['count'] = nvdiff.faces.shape[0]
        metrics['size'] = mesh_size(nvdiff.vertices, nvdiff.faces)
        metrics['cratio'] = ref_size/metrics['size']

        evaluations.setdefault('nvdiffmodeling', []).append(metrics)

    os.makedirs('evals', exist_ok=True)
    torch.save(evaluations, os.path.join('evals', name + '.pt'))

def ngf_mesh(ngf, rate=16, reduce=True) -> Mesh:
    with torch.no_grad():
        uvs = ngf.sample_uniform(rate)
        V = ngf.eval(*uvs)
        base = ngf.base(rate)

    cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
    remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)

    indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
    F = remap.remap_device(indices) if reduce else indices
    return mesh_from(V, F)

def tessellation_evaluation(prefix):
    print('tess prefix:', prefix)

    rdir = os.path.abspath(os.path.join(__file__, os.pardir))
    rdir = os.path.abspath(os.path.join(rdir, os.pardir))
    base = os.path.abspath(os.path.join(rdir, 'results'))

    def eval_tessellations(evaluator, ngf, name):
        rate_metrics = {}
        # for rate in [ 2, 4, 8, 12, 16, 32 ]:
        for rate in [ 2, 4, 8, 12, 16 ]:
            mesh = ngf_mesh(ngf, rate)
            metrics = evaluator.eval_metrics(mesh, name)
            rate_metrics[rate] = {
                'render'  : metrics['render'],
                'normal'  : metrics['normal'],
                'chamfer' : metrics['chamfer'],
                'ref'     : metrics['images']['render:ref'],
                'mesh'    : metrics['images']['render:mesh'],
            }

        return rate_metrics

    data = {}
    for root, _, files in os.walk(base):
        for file in files:
            if prefix in file and file.endswith('pt'):
                file = os.path.join(root, file)
                scene = os.path.basename(root)
                print('Processing scene', scene)
                reference = os.path.join(rdir, 'meshes', scene, 'target.obj')
                reference, _ = load_mesh(reference)
                evaluator = Evaluator(reference)

                ngf = torch.load(file)
                ngf = load_ngf(ngf)

                rm = eval_tessellations(evaluator, ngf, scene)
                data[scene] = rm

    print(data)
    torch.save(data, 'tessellation.pt')

    # import json
    # with open('tessellation.json', 'w') as f:
    #     json.dump(data, f)

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
        print('Evaluating', scene)

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

            metrics = evaluator.eval_metrics(ngf_mesh, scene)
            data.setdefault(scene, {})[k] = {
                'render'  : metrics['render'],
                'normal'  : metrics['normal'],
                'chamfer' : metrics['chamfer'],
                'ref'     : metrics['images']['normal:ref'],
                'mesh'    : metrics['images']['normal:mesh'],
                'size'    : size
            }

    torch.save(data, 'features.pt')

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
        V = ngf.eval(*sample).detach()

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

def frequency_evaluation():
    import glob
    import json
    import re

    ref = load_mesh('meshes/armadillo/target.obj')[0]
    evl = Evaluator(ref)

    data = {}

    pattern = re.compile(r'.*f(\d+)-losses.json')
    for file in glob.glob('results/frequencies/*.json'):
        freqs = int(pattern.match(file).group(1))
        db = json.load(open(file, 'r'))
        data[freqs] = { 'loss': db['render'] }

    redata = {}
    for k in sorted(list(data.keys())):
        redata[k] = data[k]
    data = redata

    rate = 16
    pattern = re.compile(r'.*f(\d+).pt')
    for file in glob.glob('results/frequencies/*.pt'):
        freqs = int(pattern.match(file).group(1))
        ngf = torch.load(file)
        ngf = load_ngf(ngf)
        print('ngf', ngf)

        # TODO: method
        with torch.no_grad():
            uvs = ngf.sample_uniform(rate)
            V = ngf.eval(*uvs)
            base = ngf.base(16)

        cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, 16)
        remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], 16)
        indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], 16)
        F = remap.remap_device(indices)

        ngf_mesh = mesh_from(V, F)

        metrics = evl.eval_metrics(ngf_mesh, 'armadillo')
        images = metrics['images']

        data[freqs]['images'] = {
                'ref': images['normal:ref'],
                'mesh': images['normal:mesh']
        }

        # print('metrics', metrics.keys())

    torch.save(data, 'frequencies.pt')

def ngf_to_mesh(ngf, rate=16):
    with torch.no_grad():
        uvs = ngf.sample_uniform(rate)
        V = ngf.eval(*uvs)
        base = ngf.base(rate)
    cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
    remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
    indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
    F = remap.remap_device(indices)
    return mesh_from(V, F)

def losses_evaluation(directory):
    import json

    # Evaluator
    ref = load_mesh('meshes/igea/target.obj')[0]
    evl = Evaluator(ref)

    # Looking for ordindary and chamfer
    ordinary = os.path.join(directory, 'experimental.pt')
    chamfer = os.path.join(directory, 'experimental-chamfer.pt')
    print('ordinary', ordinary)
    print('chamfer', chamfer)

    ngf_ord = load_ngf(torch.load(ordinary))
    ngf_chm = load_ngf(torch.load(chamfer))

    metrics_ord = evl.eval_metrics(ngf_to_mesh(ngf_ord))
    metrics_chm = evl.eval_metrics(ngf_to_mesh(ngf_chm))

    ordinary = os.path.join(directory, 'experimental-losses.json')
    chamfer = os.path.join(directory, 'experimental-chamfer-losses.json')

    chm_ord = json.load(open(ordinary, 'r'))['chamfer']
    chm_chm = json.load(open(chamfer, 'r'))['chamfer']

    time_ord = json.load(open(ordinary, 'r'))['time']
    time_chm = json.load(open(chamfer, 'r'))['time']

    # Conglomerate all the data
    data = {
            'loss:ord': chm_ord,
            'time:ord': time_ord,
            'loss:chm': chm_chm,
            'time:chm': time_chm,
            'render:ref': metrics_ord['images']['normal:ref'],
            'render:ord': metrics_ord['images']['normal:mesh'],
            'render:chm': metrics_chm['images']['normal:mesh'],
    }

    torch.save(data, 'loss-eval.pt')

def ingp_evaluation():
    # Load all the files
    directory = os.path.abspath('evals/ingp')

    ref, normalizer = load_mesh(os.path.join(directory, 'target.obj'))
    ingp_t11 = load_mesh(os.path.join(directory, 'ingp-t11-f8.obj'))[0]
    ingp_t12 = load_mesh(os.path.join(directory, 'ingp-t12-f4.obj'))[0]
    ingp_t13 = load_mesh(os.path.join(directory, 'ingp-t13-f2.obj'))[0]

    ngf = load_ngf(torch.load(os.path.join(directory, 'primary.pt')))
    ngf_mesh = ngf_to_mesh(ngf)

    print('ref minmax', ref.vertices.min(0)[0], ref.vertices.max(0)[0])
    print('ngf minmax', ngf_mesh.vertices.min(0)[0], ngf_mesh.vertices.max(0)[0])
    print('ingp minmax', ingp_t11.vertices.min(0)[0], ingp_t11.vertices.max(0)[0])

    evl = Evaluator(ref)

    data_ngf = evl.render_everything(ngf_mesh, 'skull')
    data_t11 = evl.render_everything(ingp_t11, 'skull', invert=True)
    data_t12 = evl.render_everything(ingp_t12, 'skull', invert=True)
    data_t13 = evl.render_everything(ingp_t13, 'skull', invert=True)

    chm_ngf = evl.eval_chamfer(ngf_mesh)['error']
    chm_t11 = evl.eval_chamfer(ingp_t11)['error']
    chm_t12 = evl.eval_chamfer(ingp_t12)['error']
    chm_t13 = evl.eval_chamfer(ingp_t12)['error']

    data = {
            'ref': data_ngf['normal:ref'],
            'ngf': {
                'image': data_ngf['normal:mesh'],
                'error': chm_ngf
            },
            'ngp11': {
                'image': data_t11['normal:mesh'],
                'error': chm_t11
            },
            'ngp12': {
                'image': data_t12['normal:mesh'],
                'error': chm_t12
            },
            'ngp13': {
                'image': data_t13['normal:mesh'],
                'error': chm_t13
            }
    }

    torch.save(data, 'ingp.pt')

def scroller_evaluation():
    ref = load_mesh('meshes/wreck/target.obj')[0]
    ngf = load_ngf(torch.load('results/wreck/experimental.pt'))
    ngf_mesh = ngf_to_mesh(ngf)

    evl = Evaluator(ref)
    print('ref, ngf', ref.vertices.shape[0], ngf_mesh.vertices.shape[0])

    return torch.save(evl.render_everything(ngf_mesh, 'wreck'), 'wreck-all.pt')

def teaser_evaluation():
    ref    = load_mesh('evals/teaser/target.obj')[0]
    qslim  = load_mesh('evals/teaser/qslim.obj')[0]
    nvdiff = load_mesh('evals/teaser/nvdiff.obj')[0]
    ingp   = load_mesh('evals/teaser/ingp.obj')[0]
    ngf    = load_ngf(torch.load('evals/teaser/primary.pt'))

    def ngf_to_mesh(ngf, rate=16, reduce=True) -> Mesh:
        with torch.no_grad():
            uvs = ngf.sample_uniform(rate)
            V = ngf.eval(*uvs)
            base = ngf.base(rate)

        cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
        remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)

        indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
        F = remap.remap_device(indices) if reduce else indices
        return mesh_from(V, F)

    ngf_mesh = ngf_to_mesh(ngf, reduce=False)

    evl = Evaluator(ref)

    COLOR_WHEEL = np.array([
        [0.880, 0.320, 0.320],
        [0.880, 0.530, 0.320],
        [0.880, 0.740, 0.320],
        [0.810, 0.880, 0.320],
        [0.600, 0.880, 0.320],
        [0.390, 0.880, 0.320],
        [0.320, 0.880, 0.460],
        [0.320, 0.880, 0.670],
        [0.320, 0.880, 0.880],
        [0.320, 0.670, 0.880],
        [0.320, 0.460, 0.880],
        [0.390, 0.320, 0.880],
        [0.600, 0.320, 0.880],
        [0.810, 0.320, 0.880],
        [0.880, 0.320, 0.740],
        [0.880, 0.320, 0.530]
    ])

    COLOR_WHEEL = torch.from_numpy(COLOR_WHEEL).cuda().float()

    import nvdiffrast.torch as dr
    camera = evl.get_view('dragon').unsqueeze(0)
    pindex = torch.arange(ngf.complexes.shape[0]).repeat_interleave(450)
    colors = COLOR_WHEEL[pindex % COLOR_WHEEL.shape[0]]
    patches = evl.renderer.render_false_coloring(ngf_mesh.vertices, ngf_mesh.normals, ngf_mesh.faces, colors, camera)[0]

    import matplotlib.pyplot as plt
    plt.imshow(patches.cpu().numpy())
    plt.show()

    qslim_data  = evl.eval_metrics(qslim, 'dragon')
    nvdiff_data = evl.eval_metrics(nvdiff, 'dragon')
    ingp_data   = evl.eval_metrics(ingp, 'dragon', invert=True)
    ngf_data    = evl.eval_metrics(ngf_mesh, 'dragon')

    return torch.save({
        'nrm:ref'    : ngf_data['images']['normal:ref'],
        'nrm:ngf'    : ngf_data['images']['normal:mesh'],
        'nrm:qslim'  : qslim_data['images']['normal:mesh'],
        'nrm:nvdiff' : nvdiff_data['images']['normal:mesh'],
        'nrm:ingp'   : ingp_data['images']['normal:mesh'],
        'patches'    : patches,

        'chamfer:ngf'    : ngf_data['chamfer'],
        'chamfer:qslim'  : qslim_data['chamfer'],
        'chamfer:nvdiff' : nvdiff_data['chamfer'],
        'chamfer:ingp'   : ingp_data['chamfer'],
    }, 'teaser.pt')

    # return torch.save(evl.render_everything(ngf_mesh, 'dragon'), 'teaser.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=str, help='path to reference mesh')
    parser.add_argument('--alt', type=str, help='path to alternative reference mesh')
    parser.add_argument('--results', type=str, help='path to results directory')
    parser.add_argument('--tess-prefix', type=str, help='prefix for tessellation evaluations')
    parser.add_argument('--feat-prefix', type=str, help='prefix for feature size evaluations')
    parser.add_argument('--multichart', action='store_true', help='perform multichart geometry images ablations')
    parser.add_argument('--freq', action='store_true', help='perform frequency ablations')
    parser.add_argument('--losses', type=str, help='perform loss ablations')
    parser.add_argument('--ingp', action='store_true', help='perform INGP comparisons')
    parser.add_argument('--scroller', action='store_true', help='evaluating scoller images')
    parser.add_argument('--teaser', action='store_true', help='evaluating scoller images')
    args = parser.parse_args()

    if args.results:
        scene_evaluations(args.reference, args.results, args.alt)
    elif args.tess_prefix:
        tessellation_evaluation(args.tess_prefix)
    elif args.feat_prefix:
        feature_evaluation(args.feat_prefix)
    elif args.multichart:
        multichart_evaluations()
    elif args.freq:
        frequency_evaluation()
    elif args.losses:
        losses_evaluation(args.losses)
    elif args.ingp:
        ingp_evaluation()
    elif args.scroller:
        scroller_evaluation()
    elif args.teaser:
        teaser_evaluation()
    else:
        raise NotImplementedError
