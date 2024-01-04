import os
import torch
import optext
import argparse
import imageio
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

def scene_evaluations(reference, directory, alt=None):
    import re
    import sys
    import json
    import subprocess

    ref, _ = load_mesh(reference)
    key = os.path.basename(directory)
    pattern = re.compile('lod[1-4].pt$')
    assert len(key) > 0

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

    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            if not pattern.match(file):
                continue

            print('Loading NGF from ', file)

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

            metrics = evaluator.eval_metrics(ngf_mesh)
            metrics['count'] = ngf.complexes.shape[0]
            metrics['size'] = size
            del metrics['images']

            print('  > metrics:', metrics)

            evaluations.setdefault('Ours', []).append(metrics)

    # QSlim and nvdiffmodeling at various sizes
    for size_kb in [ 100, 250, 500, 1000 ]:
        # QSlim
        qslim_search(1024 * size_kb)

        smashed, _ = load_mesh(qslim_result)

        metrics = evaluator.eval_metrics(smashed)
        metrics['count'] = smashed.faces.shape[0]
        metrics['size'] = mesh_size(smashed.vertices, smashed.faces)
        del metrics['images']

        print('  > metrics:', metrics)

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
                'out_dir' : os.path.join('evals', 'nvdiffmodeling', key)
        }

        print('running with config', data)

        PYTHON = sys.executable
        SCRIPT = os.environ['HOME'] + '/sources/nvdiffmodeling/train.py'

        os.makedirs('evals/nvdiffmodeling', exist_ok=True)
        config = os.path.join('evals', 'nvdiffmodeling', key + '.json')
        with open(config, 'w') as f:
            json.dump(data, f, indent=4)

        cmd = '{} {} --config {}'.format(PYTHON, SCRIPT, config)
        print('cmd', cmd)

        subprocess.run(cmd.split())

        nvdiff, _ = load_mesh(os.path.join('evals', 'nvdiffmodeling', key, 'mesh', 'mesh.obj'))

        metrics = evaluator.eval_metrics(nvdiff)
        metrics['count'] = nvdiff.faces.shape[0]
        metrics['size'] = mesh_size(nvdiff.vertices, nvdiff.faces)
        del metrics['images']

        print('  > metrics:', metrics)

        evaluations.setdefault('nvdiffmodeling', []).append(metrics)

    os.makedirs('evals', exist_ok=True)
    with open(os.path.join('evals', key + '.json'), 'w') as f:
        json.dump(evaluations, f)

    # TODO: store the images in evals/
    # torch.save(evaluations, os.path.join('evals', key + '.pt'))

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

def packed_geometry_image():
    import re

    scene_metrics = {}
    evaluators = {}

    p = re.compile('lod[1-4].pt$')
    for root, _, files in os.walk('.'):
        for file in files:
            if not p.match(file):
                continue

            ngf = os.path.join(root, file)
            ngf = torch.load(ngf)
            ngf = load_ngf(ngf)

            if ngf.ffin != 83:
                continue

            count = ngf.complexes.shape[0]
            print('Found file', root, file)
            scene = os.path.basename(root)
            scene_ref = os.path.join('meshes', scene, 'target.obj')
            print('  > scene %s, ref %s' % (scene, scene_ref))
            ref, _ = load_mesh(scene_ref)

            if scene != 'nefertiti':
                continue

            eval = None
            if scene_ref in evaluators:
                eval = evaluators[scene_ref]
            else:
                eval = Evaluator(ref)
                evaluators[scene_ref] = eval

            size = ngf_size(ngf)
            print('  > NGF size', size)

            maps = {}
            for rate in [ 4, 8, 12, 16 ]:
                uvs = ngf.sample_uniform(rate)
                V = ngf.eval(*uvs).detach()
                base = ngf.base(rate).detach()
                cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
                remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
                indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
                F = remap.remap_device(indices)
                ngf_mesh = mesh_from(V, F)
                # msize = mesh_size(ngf_mesh.vertices, ngf_mesh.faces)
                msize = ngf_mesh.vertices.numel() * ngf_mesh.vertices.element_size()
                metrics = eval.eval_metrics(ngf_mesh)
                del metrics['images']

                maps[rate] = (msize, metrics['render'], metrics['normal'], metrics['chamfer'])

            print('  >', maps)

            scene_metrics.setdefault(scene, {})
            scene_metrics[scene][count] = (size, maps[16][0], maps[16][-1])

    # print(scene_metrics)

    import matplotlib.pyplot as plt

    for scene, data in scene_metrics.items():
        print('scene {} -> data {}'.format(scene, data))
        patches = []
        patches_size_ngf = []
        patches_size_gim = []
        for p, metrics in data.items():
            patches.append(p)
            patches_size_ngf.append(metrics[0])
            patches_size_gim.append(metrics[1])

        plt.plot(patches, patches_size_ngf, label='NGF')
        plt.plot(patches, patches_size_gim, label='GIM')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=str, help='path to reference mesh')
    parser.add_argument('--alt', type=str, help='path to alternative reference mesh')
    parser.add_argument('--results', type=str, help='path to results directory')
    parser.add_argument('--tess-prefix', type=str, help='prefix for tessellation evaluations')
    parser.add_argument('--feat-prefix', type=str, help='prefix for feature size evaluations')
    parser.add_argument('--packed-gimg', type=str, help='path to ngf')
    args = parser.parse_args()

    if args.results:
        scene_evaluations(args.reference, args.results, args.alt)
    elif args.tess_prefix:
        tessellation_evaluation(args.tess_prefix)
    elif args.feat_prefix:
        feature_evaluation(args.feat_prefix)
    elif args.packed_gimg:
        packed_geometry_image()
    else:
        raise NotImplementedError
