# Packages for training
import imageio.v2 as imageio
import meshio
import sys
import torch
import json
import optext
import tqdm

from typing import Callable

from util import make_cmap, quadify
from ngf import NGF
from mesh import load_mesh

def load_patches(path, normalizer):
    mesh = meshio.read(path)

    v = torch.from_numpy(mesh.points).float().cuda()
    f = torch.from_numpy(mesh.cells_dict['quad']).int().cuda()
    v = normalizer(v)

    return v, f

def construct_ngf(path: str, config: dict, normalizer: Callable) -> NGF:
    # TODO: try random feature initialization
    points, complexes = load_patches(path, normalizer)
    features          = torch.zeros((points.shape[0], config['features']), device='cuda')

    points.requires_grad   = True
    features.requires_grad = True

    return NGF(points, complexes, features, config)

def tonemap_srgb(f):
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4) * 1.055 - 0.055, 12.92 * f)

def alpha_blend(img):
    alpha = img[..., 3:]
    return img[..., :3] * alpha + (1.0 - alpha)

if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Usage: python train.py <config>'

    config = sys.argv[1]
    print('Using configuration: {}'.format(config))

    # Load configuration
    with open(config, 'r') as f:
        config = json.load(f)

    # Load mesh
    target_path    = config['target']
    result_path     = config['directory']
    clusters        = config['clusters']    # TODO: or defined per experiment
    batch_size      = config['batch_size']
    resolution      = config['resolution']
    experiments     = config['experiments']

    print('Loading target mesh from {}'.format(target_path))
    target, normalizer = load_mesh(target_path)
    print('target: {} vertices, {} faces'.format(target.vertices.shape[0], target.faces.shape[0]))

    # Iterate over experiments
    for experiment in experiments:
        name = experiment['name']
        source_path = experiment['source']

        # Create a combined dict, overriding with experiment-specific values
        local_config = dict(config)
        local_config.update(experiment)

        # TODO: display config
        print('Starting experiment ', name)
        # print(local_config)

        ngf = construct_ngf(source_path, local_config, normalizer)

        def display(rate):
            import polyscope as ps

            ps.init()
            ps.register_surface_mesh('target mesh', target.vertices.cpu().numpy(), target.faces.cpu().numpy())

            base = ngf.sample(rate)['points'].detach()
            cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)

            V = ngf.eval(rate).detach()
            indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
            F = remap.remap_device(indices)

            ps.register_surface_mesh('mesh', V.cpu().numpy(), F.cpu().numpy())
            ps.show()

        def ngf_faces(rate):
            base = ngf.sample(rate)['points'].detach()
            cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)

            V = ngf.eval(rate).detach()
            indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
            F = remap.remap_device(indices)

            return F

        from tqdm import trange

        import matplotlib.pyplot as plt
        import seaborn as sns

        from kaolin.metrics.pointcloud import chamfer_distance
        from kaolin.metrics.trianglemesh import point_to_mesh_distance
        from kaolin.metrics.trianglemesh import average_edge_length

        sns.set()

        # TODO: parameters() method for ngf
        opt = torch.optim.Adam(list(ngf.mlp.parameters()) + [ ngf.points, ngf.features ], lr=1e-3)

        losses = {}

        target_face_vertices = target.vertices[target.faces]

        opt = torch.optim.Adam(list(ngf.mlp.parameters()) + [ ngf.points, ngf.features ], lr=1e-3)
        for rate in [ 4, 8, 16 ]:
            V = ngf.eval(rate).detach().unsqueeze(0)
            F = ngf_faces(rate)
            edge = average_edge_length(V, F).mean().item()

            base = ngf.sample(rate)['points'].detach()
            cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
            quads = torch.from_numpy(quadify(ngf.complexes.shape[0], rate)).int()
            vgraph = optext.vertex_graph(remap.remap(quads))

            for _ in trange(1_000):
                ngf_vertices = ngf.eval(rate)

                chamfer_loss = chamfer_distance(ngf_vertices.unsqueeze(0), target.vertices.unsqueeze(0))

                # distance, _, _ = point_to_mesh_distance(ngf_vertices.unsqueeze(0), target_face_vertices.unsqueeze(0))
                # loss = distance.mean()

                ngf_vertices_smoothed = vgraph.smooth_device(ngf_vertices, 1.0)
                smooth_loss = ((ngf_vertices - ngf_vertices_smoothed) ** 2).mean()/edge
                loss = chamfer_loss + smooth_loss/50

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.setdefault('chamfer', []).append(loss.item())
                losses.setdefault('smooth', []).append(smooth_loss.item())
                # losses.append(loss.item())

        # Clear plot
        plt.clf()
        plt.plot(losses['chamfer'], label='chamfer')
        plt.plot(losses['smooth'], label='smooth')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('chamfer distance')
        plt.yscale('log')
        plt.savefig('{}-loss.png'.format(name))

        display(16)
