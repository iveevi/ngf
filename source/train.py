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
from configurations import *

from geometry import compute_vertex_normals, compute_face_normals
from mesh import Mesh, load_mesh, simplify_mesh
from ngf import NGF
from render import Renderer, arrange_camera_views

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

    # points = points.bfloat16()
    # features = features.bfloat16()

    points.requires_grad   = True
    features.requires_grad = True

    return NGF(points, complexes, features, config)

def tonemap_srgb(f):
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4) * 1.055 - 0.055, 12.92 * f)

def alpha_blend(img):
    alpha = img[..., 3:]
    return img[..., :3] * alpha + (1.0 - alpha)

def train(target, generator, opt, sch, batch_views, iterations):
    def postprocess(img):
        img = tonemap_srgb(torch.log(torch.clamp(img, min=0, max=65535) + 1))
        return img

    def iterate(view_mats):
        V, N, F, additional = generator()

        phi = torch.rand((10, 1), device='cuda') * 2 * np.pi
        theta = torch.rand((10, 1), device='cuda') * np.pi
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        lights = torch.cat([ x, y, z ], dim=1)

        # lights = torch.rand((10, 3), device='cuda')
        # lights /= torch.linalg.norm(lights, dim=-1).unsqueeze(-1)
        # light_colors = torch.rand((10, 3), device='cuda')

        opt_imgs = renderer.render(V, N, F, lights, view_mats) #.bfloat16()
        # opt_imgs = postprocess(opt_imgs)

        with torch.no_grad():
            ref_imgs = renderer.render(target.vertices, target.normals, target.faces, lights, view_mats) #.bfloat16()
            # ref_imgs = postprocess(ref_imgs)

        # import matplotlib.pyplot as plt
        #
        # fig = plt.figure(layout='constrained')
        # subfigs = fig.subfigures(1, 2)
        #
        # subfigs[0].suptitle('References')
        # axs = subfigs[0].subplots(len(view_mats), 1)
        # for ax, img in zip(axs, ref_imgs):
        #     ax.imshow(img[..., 0].float().detach().cpu().numpy())
        #     ax.axis('off')
        #
        # subfigs[1].suptitle('References')
        # axs = subfigs[1].subplots(len(view_mats), 1)
        # for ax, img in zip(axs, opt_imgs):
        #     ax.imshow(img[..., 0].float().detach().cpu().numpy())
        #     ax.axis('off')
        #
        # plt.show()

        # Compute losses
        render_loss = (opt_imgs - ref_imgs).abs().mean()
        assert not torch.isnan(render_loss).any()

        loss = render_loss
        if additional is not None:
            loss += additional

        return loss

    losses = []
    for _ in tqdm.trange(iterations):
        loss_average = 0
        for view_mats in batch_views:
            loss = iterate(view_mats)

            # Optimization step
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

            if sch is not None:
                sch.step()

            loss_average += loss.item()

        loss_average /= len(batch_views)

        global writer, step
        writer.add_scalar('Loss', loss_average, step)
        writer.flush()
        step += 1

    return losses

def vertex_normals(V, F):
    Fn = compute_face_normals(V, F)
    N  = compute_vertex_normals(V, F, Fn)
    return N

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
        return V, N, F, None

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

    cmap   = make_cmap(ngf.complexes, ngf.points.float().detach(), base.float(), rate)
    remap  = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
    quads  = torch.from_numpy(quadify(ngf.complexes.shape[0], rate)).int()
    vgraph = optext.vertex_graph(remap.remap(quads))
    # factor = 0.5 ** 1e-3
    delta  = 1/rate

    def generator():
        nonlocal vgraph, delta

        # torch.cuda.empty_cache()
        for p in ngf.parameters():
            p.grad = None

        # base = ngf.base(rate).float()
        uvs = ngf.sampler(rate)
        V = ngf.eval(*uvs).float()

        assert not torch.isnan(V).any()

        indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
        F = remap.remap_device(indices)

        N = None
        if ngf.normals == 'numerical':
            N = ngf.eval_normals(*uvs, delta).float()
        elif ngf.normals == 'geometric':
            N = vertex_normals(V, F)
        else:
            raise NotImplementedError

        V_smoothed = vgraph.smooth_device(V, 1.0)
        laplacian_loss = laplacian_strength * (V - V_smoothed).abs().mean()

        return V, N, F, laplacian_loss

    batch_views = torch.split(views, batch_size, dim=0)
    return train(target, generator, opt, sch, batch_views, iterations)

def visualize_views(views):
    import matplotlib.pyplot as plt

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

    all_views = arrange_camera_views(target)
    add_views = arrange_views(target, 100)
    all_views = torch.concat([all_views, add_views])

    # all_views = arrange_views(target, 200)

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

        losses = []
        laplacian_strength = 1.0

        def display(rate):
            import polyscope as ps

            ps.init()
            ps.register_surface_mesh('target mesh', target.vertices.cpu().numpy(), target.faces.cpu().numpy())

            with torch.no_grad():
                base = ngf.base(rate).float()
                uvs = ngf.sampler(rate)
                V = ngf.eval(*uvs).float()

            cmap = make_cmap(ngf.complexes, ngf.points.float().detach(), base, rate)
            remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
            indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
            F = remap.remap_device(indices)

            ps.register_surface_mesh('mesh', V.cpu().numpy(), F.cpu().numpy())
            ps.register_surface_mesh('base', base.cpu().numpy(), F.cpu().numpy())
            ps.show()

        rate = 4
        while rate <= resolution:
            torch.cuda.empty_cache()
            # for group in opt.param_groups:
            #     group['lr'] = 1e-3

            opt = torch.optim.Adam(list(ngf.mlp.parameters()) + [ ngf.points, ngf.features ], 1e-3)
            sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)

            print('Refining with rate {}'.format(rate))
            losses += refine(target, ngf, rate, views, laplacian_strength, opt, sch, iterations=250)

            laplacian_strength *= 0.75
            rate *= 2

        # display(16)

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

        del ngf
