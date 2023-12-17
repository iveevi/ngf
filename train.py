# Packages for training
import imageio.v2 as imageio
import meshio
import os
import sys
import torch
import json
import optext

from dataclasses import dataclass
from scripts.render import NVDRenderer
from scripts.geometry import compute_vertex_normals, compute_face_normals
from tqdm import trange

from mlp import *
from util import *
from configurations import *

from ngf import NGF
from mesh import Mesh, load_mesh, simplify_mesh

def arrange_views(simplified: Mesh, cameras: int):
    seeds = list(torch.randint(0, simplified.faces.shape[0], (cameras,)).numpy())
    clusters = optext.cluster_geometry(simplified.optg, seeds, 3)

    cluster_centroids = []
    cluster_normals = []

    for cluster in clusters:
        faces = simplified.faces[cluster]

        v0 = simplified.vertices[faces[:, 0]]
        v1 = simplified.vertices[faces[:, 1]]
        v2 = simplified.vertices[faces[:, 2]]
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

def construct_renderer():
    environment = imageio.imread('media/images/environment.hdr', format='HDR-FI')
    environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
    alpha       = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
    environment = torch.cat((environment, alpha), dim=-1)

    # TODO: use trimmed fovs in various views to cut down on wasted pixels
    scene_parameters = {}
    scene_parameters['res_x']        = 1280
    scene_parameters['res_y']        = 720
    scene_parameters['fov']          = 45.0
    scene_parameters['near_clip']    = 0.1
    scene_parameters['far_clip']     = 1000.0
    scene_parameters['envmap']       = environment
    scene_parameters['envmap_scale'] = 1.0

    # TODO: refactoring...
    return NVDRenderer(scene_parameters, shading=True, boost=3)

def load_patches(path, normalizer):
    mesh = meshio.read(path)

    v = torch.from_numpy(mesh.points).float().cuda()
    f = torch.from_numpy(mesh.cells_dict['quad']).int().cuda()
    v = normalizer(v)

    return v, f

def construct_ngf(path, config, normalizer):
    # TODO: try random feature initialization
    points, complexes = load_patches(path, normalizer)
    features = torch.zeros((points.shape[0], config.features), device='cuda')

    points.requires_grad = True
    features.requires_grad = True

    mlp = MLP(config).cuda()

    return NGF(points, complexes, features, mlp)

def tonemap_srgb(f):
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4) * 1.055 - 0.055, 12.92 * f)

def alpha_blend(img):
    alpha = img[..., 3:]
    return img[..., :3] * alpha + (1.0 - alpha)

def kickoff(target, ngf, views, batch_size):
    from largesteps.geometry import compute_matrix
    from largesteps.optimize import AdamUniform
    from largesteps.parameterize import from_differential, to_differential

    base, _ = ngf.sample(4)
    base = base.detach()

    base_indices = shorted_indices(base.cpu().numpy(), ngf.complexes.cpu().numpy(), 4)
    tch_base_indices = torch.from_numpy(base_indices).int()

    cmap  = make_cmap(ngf.complexes, ngf.points.detach(), base, 4)
    remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], 4)

    F     = remap.remap(tch_base_indices).cuda()
    indices = quadify(ngf.complexes.cpu().numpy(), 4)

    steps     = 1_000  # Number of optimization steps
    step_size = 0.1    # Step size
    lambda_   = 10     # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_ * L)

    # Optimization setup
    M = compute_matrix(base, F, lambda_)
    U = to_differential(M, base)

    U.requires_grad = True
    opt = AdamUniform([ U ], step_size)

    indices = quadify(ngf.complexes.cpu().numpy(), 4)
    vgraph = optext.vertex_graph(F.cpu())
    cgraph = optext.conformal_graph(torch.from_numpy(indices).int())
    # print('vgraph:', vgraph, 'cgraph:', cgraph)

    a, opp_a = cgraph[:, 0], cgraph[:, 1]
    b, opp_b = cgraph[:, 2], cgraph[:, 3]
    print('a:', a.shape, 'opp_a:', opp_a.shape)
    print('b:', b.shape, 'opp_b:', opp_b.shape)

    # TODO: wrapper for inverse rendering (taking a functional and optimizer...)

    # Optimization loop
    # batch_size = batch(sample_rate)
    for _ in trange(steps):
        # Batch the views into disjoint sets
        assert len(views) % batch_size == 0
        batch_views = torch.split(views, batch_size, dim=0)
        for view_mats in batch_views:
            V = from_differential(M, U, 'Cholesky')

            Fn = compute_face_normals(V, F)
            N = compute_vertex_normals(V, F, Fn)

            # TODO: custom override
            opt_imgs = renderer.render(V, N, F, view_mats)
            ref_imgs = renderer.render(target.vertices, target.normals, target.faces, view_mats)

            opt_imgs = alpha_blend(opt_imgs)
            ref_imgs = alpha_blend(ref_imgs)

            opt_imgs = tonemap_srgb(torch.log(torch.clamp(opt_imgs, min=0, max=65535) + 1))
            ref_imgs = tonemap_srgb(torch.log(torch.clamp(ref_imgs, min=0, max=65535) + 1))

            V_smoothed = vgraph.smooth_device(V, 1.0)
            V_smoothed = vgraph.smooth_device(V_smoothed, 1.0)

            # Compute losses
            # TODO: tone mapping from NVIDIA paper
            render_loss = (opt_imgs - ref_imgs).abs().mean()
            laplacian_loss = (V - V_smoothed).abs().mean()
            loss = render_loss + laplacian_loss

            # Optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()

    V = from_differential(M, U, 'Cholesky')
    V = V.detach()

    # import polyscope as ps
    # ps.init()
    # ps.register_surface_mesh('target', target.vertices.cpu().numpy(), target.faces.cpu().numpy())
    # ps.register_surface_mesh('base', base.detach().cpu().numpy(), indices)
    # ps.register_surface_mesh('kickoff', V.cpu().numpy(), F.cpu().numpy())
    # ps.show()

    # Overfit to this result
    opt = AdamUniform(list(ngf.mlp.parameters()) + [ ngf.points, ngf.features ], 1e-3)

    # TODO: custom label
    for _ in trange(1000):
        lerped_points, lerped_features = ngf.sample(4)
        # print('lerped_points:', lerped_points.shape, 'lerped_features:', lerped_features.shape)
        # print('devices:', lerped_points.device, lerped_features.device)
        V = ngf.mlp(points=lerped_points, features=lerped_features)
        loss = (V - base).abs().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    # V = V.detach()
    # ps.register_surface_mesh('overfit', V.cpu().numpy(), F.cpu().numpy())
    # ps.show()

    return opt

    # TODO: gather training data...

# TODO: kwargs
def refine(target, ngf, rate, views, laplacian_strength, opt, iterations):
    # opt      = torch.optim.Adam(list(m.parameters()) + [ features, points ], lr)
    # base, _  = sample(complexes, points, features, sample_rate)
    base, _  = ngf.sample(rate)
    cmap     = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
    remap    = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
    vgraph   = None

    for i in trange(iterations):
        # Batch the views into disjoint sets
        assert len(views) % batch_size == 0
        batch_views = torch.split(views, batch_size, dim=0)

        if i % 10 == 0:
            # print('Rebuilding vgraph...')
            lerped_points, lerped_features = ngf.sample(rate)
            V = ngf.mlp(points=lerped_points, features=lerped_features)
            indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
            F = remap.remap_device(indices)
            vgraph = optext.vertex_graph(F.cpu())

        render_loss_sum = 0
        for view_mats in batch_views:
            lerped_points, lerped_features = ngf.sample(rate)
            V = ngf.mlp(points=lerped_points, features=lerped_features)

            indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
            F = remap.remap_device(indices)

            Fn = compute_face_normals(V, F)
            N = compute_vertex_normals(V, F, Fn)

            opt_imgs = renderer.render(V, N, F, view_mats)
            ref_imgs = renderer.render(target.vertices, target.normals, target.faces, view_mats)

            opt_imgs = alpha_blend(opt_imgs)
            ref_imgs = alpha_blend(ref_imgs)

            opt_imgs = tonemap_srgb(torch.log(torch.clamp(opt_imgs, min=0, max=65535) + 1))
            ref_imgs = tonemap_srgb(torch.log(torch.clamp(ref_imgs, min=0, max=65535) + 1))

            V_smoothed = vgraph.smooth_device(V, 1.0)
            V_smoothed = vgraph.smooth_device(V_smoothed, 1.0)

            # Compute losses
            # TODO: tone mapping from NVIDIA paper
            render_loss = (opt_imgs - ref_imgs).abs().mean()
            laplacian_loss = (V - V_smoothed).abs().mean()
            loss = render_loss + laplacian_strength * laplacian_loss

            render_loss_sum += render_loss.item()

            # Optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()

    # import polyscope as ps
    # ps.init()
    # ps.remove_all_structures()
    # ps.register_surface_mesh('refined', V.detach().cpu().numpy(), F.cpu().numpy())
    # ps.register_surface_mesh('base', base.detach().cpu().numpy(), indices.cpu().numpy())
    # ps.register_surface_mesh('target', target.vertices.cpu().numpy(), target.faces.cpu().numpy())
    # ps.show()

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
    # iterations    = config['iterations']
    batch_size      = config['batch_size']
    resolution      = config['resolution']
    encoding_levels = config['encoding_levels']
    experiments     = config['experiments']

    print('Loading target mesh from {}'.format(target_path))
    target, normalizer = load_mesh(target_path)
    print('target: {} vertices, {} faces'.format(target.vertices.shape[0], target.faces.shape[0]))

    simplified = simplify_mesh(target, 10_000, normalizer)
    print('simplified: {} vertices, {} faces'.format(simplified.vertices.shape[0], simplified.faces.shape[0]))

    # Generate cameras
    print('geometry of simplified: {}'.format(simplified.optg))
    views = arrange_views(simplified, clusters)
    print('views: {} cameras'.format(views.shape[0]))

    # Iterate over experiments
    for experiment in experiments:
        print('Starting experiment {}'.format(experiment))

        name = experiment['name']
        source_path = experiment['source']
        features = experiment['features']

        config = Configuration(features=features, encoding_levels=encoding_levels)

        # TODO: pass rest of the options to the mlp
        ngf = construct_ngf(source_path, config, normalizer)

        print('ngf: {} points, {} features'.format(ngf.complexes.shape[0], ngf.features.shape[1]))

        renderer = construct_renderer()
        print('renderer:', renderer)

        # TODO: return the optimizer
        opt = kickoff(target, ngf, views, batch_size)

        # opt = torch.optim.Adam(list(ngf.mlp.parameters()) + [ ngf.points, ngf.features ], 1e-3)
        laplacian_strength = 1.0

        rate = 4
        while rate <= resolution:
            print('Refining with rate {}'.format(rate))
            refine(target, ngf, rate, views, laplacian_strength, opt, iterations=100 * rate)
            laplacian_strength *= 0.75
            rate *= 2

        os.makedirs(result_path, exist_ok=True)
        result = os.path.join(result_path, name + '.pt')
        print('Saving result to {}'.format(result))
        ngf.save(result)
