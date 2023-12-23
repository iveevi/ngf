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
    clusters = optext.cluster_geometry(mesh.optg, seeds, 3)

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
            ref_imgs = self.renderer.render(self.reference.vertices, self.reference.normals, self.reference.faces, batch_views)
            mesh_imgs = self.renderer.render(mesh.vertices, mesh.normals, mesh.faces, batch_views)

            error = torch.mean(torch.abs(ref_imgs - mesh_imgs))
            errors.append(error.item())

        # Present view
        # TODO: should be configurable
        eye    = torch.tensor([ 0, 0, -2.5 ], device='cuda')
        up     = torch.tensor([0.0, -1.0, 0.0], device='cuda')
        center = torch.tensor([0.0, 0.0, 0.0], device='cuda')

        camera = lookat(eye, center, up).unsqueeze(0)

        ref_img = self.renderer.render(self.reference.vertices, self.reference.normals, self.reference.faces, camera)[0]
        mesh_img = self.renderer.render(mesh.vertices, mesh.normals, mesh.faces, camera)[0]

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
        msize = mesh_size(mesh.vertices, mesh.faces)
        cratio = self.ref_size/msize
        return {
            'render': render['error'],
            'normal': normals['error'],
            'chamfer': chamfer['error'],
            'size': msize,
            'cratio': cratio,
            'images': {
                'render:ref': render['ref'],
                'render:mesh': render['mesh'],
                'normal:ref': normals['ref'],
                'normal:mesh': normals['mesh'],
            }
        }

def scene_evaluations(reference, directory):
    reference, normalizer = load_mesh(reference)
    key = os.path.basename(directory)

    evaluator = Evaluator(reference)

    evaluations = {}
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            if 'unpacked' in path:
                continue

            if file.endswith('.obj') and file.startswith('nvdiffmodeling'):
                print('Evaluating:', path)
                # data = evaluate_mesh(path)
                # write('Nvdiffmodeling', data)

            if not file.endswith('.pt'):
                continue

            print('Loading NGF from ', file)

            ngf = os.path.join(root, file)
            ngf = torch.load(ngf)
            ngf = load_ngf(ngf)
            print(ngf)

            size = ngf.size()
            print('  > size bytes', size)

            V = ngf.eval(16).detach()

            # TODO: util function
            base = ngf.sample(16)['points'].detach()
            cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, 16)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], 16)
            indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], 16)
            F = remap.remap_device(indices)

            ngf_mesh = mesh_from(V, F)

            metrics = evaluator.eval_metrics(ngf_mesh)
            metrics['count'] = ngf.complexes.shape[0]
            print('  > metrics:', metrics.keys())

            evaluations.setdefault('NGF (Ours)', []).append(metrics)

    os.makedirs('evals', exist_ok=True)
    torch.save(evaluations, os.path.join('evals', key + '.pt'))

def tessellation_evaluation(prefix):
    print('tess prefix:', prefix)

    rdir = os.path.abspath(os.path.join(__file__, os.pardir))
    rdir = os.path.abspath(os.path.join(rdir, os.pardir))
    base = os.path.abspath(os.path.join(rdir, 'results'))

    def eval_tessellations(evaluator, ngf):
        rate_metrics = {}
        for rate in [ 2, 4, 8, 12, 16 ]:
            mesh = ngf.mesh(rate)
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
    with open('tess.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=str, help='path to refernce mesh')
    parser.add_argument('--results', type=str, help='path to results directory')
    parser.add_argument('--prefix', type=str, help='prefix for tessellation evaluations')
    args = parser.parse_args()

    if args.results:
        scene_evaluations(args.reference, args.directory)
    elif args.prefix:
        tessellation_evaluation(args.prefix)
    else:
        raise NotImplementedError
