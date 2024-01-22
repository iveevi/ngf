# Packages for training
import imageio.v2 as imageio
import meshio
import os
import sys
import torch
import json
import optext
import tqdm

from torch.utils.tensorboard import SummaryWriter

from util import *
from geometry import compute_vertex_normals, compute_face_normals, vertex_density
from mesh import Mesh, load_mesh, simplify_mesh
from ngf import NGF
from render import Renderer, arrange_camera_views

def construct_renderer():
    path = os.path.join(os.path.dirname(__file__), '../images/environment.hdr')
    environment = imageio.imread(path, format='HDR')
    environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
    alpha       = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
    environment = torch.cat((environment, alpha), dim=-1)
    return Renderer(width=256, height=256, fov=45.0, near=0.1, far=1000.0, envmap=environment)

def load_patches(path, normalizer):
    mesh = meshio.read(path)
    v = torch.from_numpy(mesh.points).float().cuda()
    f = torch.from_numpy(mesh.cells_dict['quad']).int().cuda()
    v = normalizer(v)
    return v, f

def construct_ngf(path: str, config: dict, normalizer) -> NGF:
    # TODO: try random feature initialization
    points, complexes = load_patches(path, normalizer)
    features          = torch.zeros((points.shape[0], config['features']), device='cuda')
    # features          = torch.randn((points.shape[0], config['features']), device='cuda')
    points.requires_grad   = True
    features.requires_grad = True
    return NGF(points, complexes, features, config)

def vertex_normals(V, F):
    Fn = compute_face_normals(V, F)
    N  = compute_vertex_normals(V, F, Fn)
    return N

def train(target, generator, opt, sch, viewset, iterations):
    import torch

    # Laplacian?
    from kaolin.metrics.pointcloud import chamfer_distance

    def lift(I):
        return torch.cat((I, I.sin(), I.cos()), dim=-1)

    views, batch_size = viewset

    VT = target.vertices[target.faces].reshape(-1, 3)
    FT = torch.arange(VT.shape[0], device=VT.device, dtype=torch.int32).reshape(-1, 3)
    NT = vertex_normals(VT, FT)

    def iterate(ref_data, view_mats):
        V, N, F, additional = generator()

        opt_imgs = renderer.render_attributes(V, N, F, view_mats)

        # import matplotlib.pyplot as plt
        # import seaborn as sns
        #
        # sns.set_theme()
        #
        # fig = plt.figure(layout='constrained')
        # subfigs = fig.subfigures(1, 2)
        #
        # subfigs[0].suptitle('References')
        # axs = subfigs[0].subplots(len(view_mats), 1)
        # for ax, img in zip(axs, ref_laps):
        #     ax.imshow(img[..., 3].detach().cpu().numpy())
        #     ax.axis('off')
        #
        # subfigs[1].suptitle('Opt')
        # axs = subfigs[1].subplots(len(view_mats), 1)
        # for ax, img in zip(axs, opt_laps):
        #     ax.imshow(img[..., 3].detach().cpu().numpy())
        #     ax.axis('off')
        #
        # plt.show()

        # Compute losses
        render_loss = (lift(opt_imgs) - ref_data).abs().mean()
        loss = render_loss
        if additional is not None:
            loss += additional

        return loss

    # Caching reference views
    cache = []
    for view_mats in views.split(batch_size):
        ref_data = renderer.render_attributes(VT, NT, FT, view_mats)
        cache.append(lift(ref_data.cpu()))

    all_ref_data = torch.concat(cache, dim=0)
    all_ref_data.pin_memory()

    losses = { 'render': [], 'chamfer': [] }
    import torch.nn.utils
    for _ in tqdm.trange(iterations):
        loss_average = 0

        indices = torch.randperm(views.shape[0])
        perm_ref_data = all_ref_data[indices].split(batch_size)
        perm_view_mats = views[indices.cuda()].split(batch_size)

        # Iterate through the batch
        for batch_ref_data, batch_views in zip(perm_ref_data, perm_view_mats):
            loss = iterate(batch_ref_data.cuda(), batch_views)

            # Optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_average += loss.item() / batch_size

        global writer, step

        with torch.no_grad():
            V = generator()[0].unsqueeze(0)
            T = target.vertices.unsqueeze(0)
            chamfer = chamfer_distance(V, T)

        losses['render'].append(loss_average)
        losses['chamfer'].append(chamfer.item())

        writer.add_scalar('Render', loss_average, step)
        writer.add_scalar('Chamfer', chamfer, step)
        writer.flush()
        step += 1

    return losses

def kickoff(target, ngf, views, batch_size):
    from largesteps.geometry import compute_matrix
    from largesteps.optimize import AdamUniform
    from largesteps.parameterize import from_differential, to_differential

    with torch.no_grad():
        base = ngf.base(4).float()

    base_indices     = shorted_indices(base.cpu().numpy(), ngf.complexes.cpu().numpy(), 4)
    tch_base_indices = torch.from_numpy(base_indices).int()

    cmap  = make_cmap(ngf.complexes, ngf.points.float().detach(), base, 4)
    remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], 4)
    F     = remap.remap(tch_base_indices).cuda()

    steps     = 1_00
    step_size = 1e-3

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
        return V, N, F, None

    # batch_views = torch.split(views, batch_size, dim=0)
    train(target, generator, opt, None, (views, batch_size), steps)

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

def refine(target, ngf, rate, views, opt, sch, iterations):
    base = ngf.base(rate).detach()

    cmap   = make_cmap(ngf.complexes, ngf.points.float().detach(), base.float(), rate)
    remap  = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
    quads  = torch.from_numpy(quadify(ngf.complexes.shape[0], rate)).int()
    vgraph = optext.vertex_graph(remap.remap(quads))
    delta  = 1/rate

    def generator():
        nonlocal vgraph, delta

        for p in ngf.parameters():
            p.grad = None

        uvs = ngf.sampler(rate)
        V = ngf.eval(*uvs)

        indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
        F = remap.remap_device(indices)
        N = vertex_normals(V, F)

        V_smoothed = vgraph.smooth_device(V, 1.0)
        laplacian_loss = (V - V_smoothed).abs().mean()

        return V, N, F, laplacian_loss

    return train(target, generator, opt, sch, (views, batch_size), iterations)

def visualize_views(views):
    import matplotlib.pyplot as plt

    views = views[:4]
    N = int(np.sqrt(len(views)))
    M = (len(views) + N - 1) // N

    fig, axs = plt.subplots(N, M)
    axs = axs.flatten()

    renderer = construct_renderer()
    for ax, view in zip(axs, views):
        image = renderer.render_normals(target.vertices, target.normals, target.faces, view.unsqueeze(0))[0]
        ax.imshow(image.cpu().numpy())
        ax.axis('off')

    plt.show()

if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Usage: python train.py <config>'

    config = sys.argv[1]
    print('Using configuration: {}'.format(config))

    # Load configuration
    with open(config, 'r') as f:
        config = json.load(f)

    # Load mesh
    target_path = config['target']
    result_path = config['directory']
    clusters    = config['clusters']
    batch_size  = config['batch_size']
    resolution  = config['resolution']
    experiments = config['experiments']
    scene       = os.path.basename(os.path.dirname(target_path))
    print('Scene', scene)

    global_batch_size = batch_size

    target, normalizer = load_mesh(target_path)

    print('Loaded target: {} vertices, {} faces'.format(target.vertices.shape[0], target.faces.shape[0]))

    # all_views = arrange_camera_views(target)
    # add_views = arrange_views(target, 100)
    # all_views = torch.concat([all_views, add_views])
    all_views = arrange_views(target, 200)
    visualize_views(all_views)

    # Iterate over experiments
    for experiment in experiments:
        torch.cuda.empty_cache()

        name = experiment['name']
        source_path = experiment['source']

        title = '%s-%s' % (scene, name)

        writer = SummaryWriter(os.path.join('runs', title))
        step = 0

        if 'batch_size' in experiment:
            batch_size = experiment['batch_size']
        else:
            batch_size = global_batch_size

        cut_size = batch_size * (len(all_views) // batch_size)
        views = all_views[:cut_size]

        # Create a combined dict, overriding with experiment-specific values
        local_config = dict(config)
        local_config.update(experiment)

        print('Starting experiment ', name)
        print(local_config)

        ngf = construct_ngf(source_path, local_config, normalizer)
        print('# of complexes', ngf.complexes.shape[0])
        print('# of cut views', cut_size)

        renderer = construct_renderer()
        kickoff(target, ngf, views, batch_size)

        laplacian_strength = 1.0

        def display(rate):
            import polyscope as ps

            ps.init()
            ps.register_surface_mesh('target mesh', target.vertices.cpu().numpy(), target.faces.cpu().numpy())

            with torch.no_grad():
                base = ngf.base(rate).float()
                uvs = ngf.sample_uniform(rate)
                V = ngf.eval(*uvs).float()

            cmap = make_cmap(ngf.complexes, ngf.points.float().detach(), base, rate)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
            indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
            F = remap.remap_device(indices)

            ps.register_surface_mesh('mesh', V.cpu().numpy(), F.cpu().numpy())
            ps.register_surface_mesh('base', base.cpu().numpy(), F.cpu().numpy())
            ps.show()

        losses = { 'render': [], 'chamfer': [] }

        for rate in [ 4, 8, 16 ]:
            torch.cuda.empty_cache()
            opt = torch.optim.Adam(list(ngf.mlp.parameters()) + [ ngf.points, ngf.features ], 1e-3)
            sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)

            print('Refining with rate {}'.format(rate))
            new_losses = refine(target, ngf, rate, views, opt, None, iterations=250)
            # display(rate)

            losses['render'] += new_losses['render']
            losses['chamfer'] += new_losses['chamfer']

        display(16)

        # Saving data
        os.makedirs(result_path, exist_ok=True)
        result = os.path.join(result_path, name + '.pt')
        print('Saving result to {}'.format(result))
        ngf.save(result)

        fout = os.path.join(result_path, name + '-losses.json')
        print('Saving losses to', fout)
        with open(os.path.join(result_path, name + '-losses.json'), 'w') as fout:
            json.dump(losses, fout)

        del ngf
