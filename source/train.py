# Packages for training
import imageio.v2 as imageio
import meshio
import os
import sys
import torch
import json
import optext
import tqdm

from render import Renderer
from geometry import compute_vertex_normals, compute_face_normals

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
    path = os.path.join(os.path.dirname(__file__), '../images/environment.hdr')
    environment = imageio.imread(path, format='HDR-FI')
    environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
    alpha       = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
    environment = torch.cat((environment, alpha), dim=-1)

    return Renderer(width=1024, height=1024, fov=45.0, near=0.1, far=1000.0, envmap=environment)

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

def train(target, generator, opt, sch, batch_views, iterations, laplacian_strength=1.0):
    def postprocess(img):
        img = tonemap_srgb(torch.log(torch.clamp(img, min=0, max=65535) + 1))
        return img

    def iterate(view_mats, ref_imgs):
        V, N, F, aV, vgraph = generator()

        opt_imgs = renderer.render(V, N, F, view_mats)
        opt_imgs = postprocess(opt_imgs)

        # Compute losses
        render_loss    = (opt_imgs - ref_imgs).abs().mean()

        V_smoothed = vgraph.smooth_device(aV, 1.0)
        laplacian_loss = (aV - V_smoothed).abs().mean()

        return render_loss + laplacian_strength * laplacian_loss

    bar = tqdm.trange(iterations, desc='Training, loss: inf')

    ref_imgs_list = []
    for view_mats in batch_views:
        ref_imgs = renderer.render(target.vertices, target.normals, target.faces, view_mats)
        ref_imgs = postprocess(ref_imgs)
        ref_imgs_list.append(ref_imgs)

    losses = []
    for _ in bar:
        avg = 0.0
        for view_mats, ref_imgs in zip(batch_views, ref_imgs_list):
            loss = iterate(view_mats, ref_imgs)

            # Optimization step
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

            if sch is not None:
                sch.step()

            avg += loss.item()

        # TODO: measure the chamfer... (with no grad...)
        avg /= len(batch_views)
        losses.append(avg)

        bar.set_description('Training, loss: {:.4f}'.format(avg))

    return losses

def vertex_normals(V, F):
    Fn = compute_face_normals(V, F)
    N  = compute_vertex_normals(V, F, Fn)
    return N

def kickoff(target, ngf, views, batch_size):
    from largesteps.geometry import compute_matrix
    from largesteps.optimize import AdamUniform
    from largesteps.parameterize import from_differential, to_differential

    base = ngf.base(4).detach()

    base_indices     = shorted_indices(base.cpu().numpy(), ngf.complexes.cpu().numpy(), 4)
    tch_base_indices = torch.from_numpy(base_indices).int()

    cmap  = make_cmap(ngf.complexes, ngf.points.detach(), base, 4)
    remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], 4)
    F     = remap.remap(tch_base_indices).cuda()

    steps     = 1_00
    step_size = 1e-2

    # Optimization setup
    M = compute_matrix(base, F, lambda_ = 100)
    U = to_differential(M, base)

    U.requires_grad = True
    opt = AdamUniform([ U ], step_size)

    quads = torch.from_numpy(quadify(ngf.complexes.shape[0], 4)).int()
    vgraph = optext.vertex_graph(remap.remap(quads))

    def generator():
        V = from_differential(M, U, 'Cholesky')
        N = vertex_normals(V, F)
        return V, N, F, V, vgraph

    batch_views = torch.split(views, batch_size, dim=0)
    train(target, generator, opt, None, batch_views, steps)

    V = from_differential(M, U, 'Cholesky')
    V = V.detach()

    # Overfit to this result
    opt = torch.optim.Adam(list(ngf.mlp.parameters()) + [ ngf.points, ngf.features ], 1e-3)

    bar = tqdm.trange(1_000, desc='Overfitting, loss: inf')
    for _ in bar:
        uvs = ngf.sample_uniform(4)
        V = ngf.eval(*uvs)

        loss = (V - base).abs().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        bar.set_description('Overfitting, loss: {:.4f}'.format(loss.item()))

def refine(target, ngf, rate, views, laplacian_strength, opt, sch, iterations):
    base = ngf.base(rate).detach()

    cmap   = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
    remap  = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
    quads  = torch.from_numpy(quadify(ngf.complexes.shape[0], rate)).int()
    vgraph = optext.vertex_graph(remap.remap(quads))
    factor = 0.5 ** 1e-3
    delta  = 1/rate

    def generator():
        nonlocal vgraph, delta

        base = ngf.base(rate)
        uvs = ngf.sampler(rate)
        V = ngf.eval(*uvs)

        indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
        F = remap.remap_device(indices)
        N = None

        # TODO: turn into a functional...
        if ngf.normals == 'numerical':
            N = ngf.eval_normals(*uvs, delta)
        elif ngf.normals == 'geometric':
            N = vertex_normals(V, F)
        else:
            raise NotImplementedError

        # delta *= factor

        return V, N, F, base, vgraph

    batch_views = torch.split(views, batch_size, dim=0)
    return train(target, generator, opt, sch, batch_views, iterations, laplacian_strength=laplacian_strength)

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
    clusters        = config['clusters']
    batch_size      = config['batch_size']
    resolution      = config['resolution']
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
        name = experiment['name']
        source_path = experiment['source']

        # Create a combined dict, overriding with experiment-specific values
        local_config = dict(config)
        local_config.update(experiment)

        print('Starting experiment ', name)
        print(local_config)

        ngf      = construct_ngf(source_path, local_config, normalizer)
        renderer = construct_renderer()
        kickoff(target, ngf, views, batch_size)

        losses = []
        laplacian_strength = 1.0

        def display(rate):
            import polyscope as ps

            ps.init()
            ps.register_surface_mesh('target mesh', target.vertices.cpu().numpy(), target.faces.cpu().numpy())

            base = ngf.base(rate).detach()
            cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)

            uvs = ngf.sampler(rate)
            V = ngf.eval(*uvs).detach()
            indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
            F = remap.remap_device(indices)

            ps.register_surface_mesh('mesh', V.cpu().numpy(), F.cpu().numpy())
            ps.register_surface_mesh('base', base.cpu().numpy(), F.cpu().numpy())
            ps.show()

        opt = torch.optim.Adam(list(ngf.mlp.parameters()) + [ ngf.points, ngf.features ], 1e-3)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9999)

        rate = 4
        while rate <= resolution:
            torch.cuda.empty_cache()
            for group in opt.param_groups:
                group['lr'] = 1e-3

            print('Refining with rate {}'.format(rate))
            losses += refine(target, ngf, rate, views, laplacian_strength, opt, sch, iterations=1000)

            display(rate)
            laplacian_strength *= 0.75
            rate *= 2

            # renderer.res = (2 * renderer.res[0], 2 * renderer.res[1])
            # batch_size //= 4

        os.makedirs(result_path, exist_ok=True)
        result = os.path.join(result_path, name + '.pt')
        print('Saving result to {}'.format(result))
        ngf.save(result)

        # Save losses as CSV
        csv_path = os.path.join(result_path, name + '-losses.csv')
        print('Saving losses to {}'.format(csv_path))
        with open(csv_path, 'w') as f:
            string = ','.join(map(str, losses))
            f.write(string)
