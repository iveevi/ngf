import torch
import numpy as np
import polyscope as ps
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from tqdm import trange

sys.path.append('..')
from mlp import *
from util import *
from configurations import *
from scripts.geometry import *

sns.set()

def flat(sample_rate=16, C=2):
    # Compute all used vertices
    super_sample_rate = C * sample_rate - (C - 1)
    full = []
    for i in range(super_sample_rate):
        for j in range(super_sample_rate):
            u, v = (i / super_sample_rate), (j / super_sample_rate)
            v = np.array([u, 0, v])
            full.append(v)

    full = np.array(full).reshape((super_sample_rate, super_sample_rate, 3))
    print('full:', full.shape)

    # Per complex points...
    points = []
    for i in range(C):
        for j in range(C):
            i0, j0 = i * sample_rate - i, j * sample_rate - j
            i1, j1 = i0 + sample_rate, j0 + sample_rate
            pts = full[i0:i1, j0:j1, :]
            print('pts:', pts.shape)
            points.append(pts)

    points = np.array(points).reshape((-1, 3))

    # Complex corners
    corners = []
    for i in range(C + 1):
        for j in range(C + 1):
            i0, j0 = i * sample_rate - i, j * sample_rate - j
            print(i0, j0)
            corners.append(full[i0, j0, :])

    corners = np.array(corners).reshape((-1, 3))

    # Complex arrangement
    complexes = []
    for i in range(C):
        for j in range(C):
            c = i * (C + 1) + j
            complexes.append([c, c + 1, c + (C + 1) + 1, c + (C + 1)])

    complexes = np.array(complexes).reshape((C * C, 4))

    print('complexes:', complexes.shape)
    print('points:', points.shape)
    print('corners:', corners.shape)

    print(complexes)

    return complexes, points, corners

def linear(sample_rate=16, C=2):
    # Compute all used vertices
    super_sample_rate = C * sample_rate - (C - 1)
    full = []
    for i in range(super_sample_rate):
        for j in range(super_sample_rate):
            u, v = (i / super_sample_rate), (j / super_sample_rate)
            v = np.array([u, u + v, v])
            full.append(v)

    full = np.array(full).reshape((super_sample_rate, super_sample_rate, 3))
    print('full:', full.shape)

    # Per complex points...
    points = []
    for i in range(C):
        for j in range(C):
            i0, j0 = i * sample_rate - i, j * sample_rate - j
            i1, j1 = i0 + sample_rate, j0 + sample_rate
            pts = full[i0:i1, j0:j1, :]
            print('pts:', pts.shape)
            points.append(pts)

    points = np.array(points).reshape((-1, 3))

    # Complex corners
    corners = []
    for i in range(C + 1):
        for j in range(C + 1):
            i0, j0 = i * sample_rate - i, j * sample_rate - j
            print(i0, j0)
            corners.append(full[i0, j0, :])

    corners = np.array(corners).reshape((-1, 3))

    # Complex arrangement
    complexes = []
    for i in range(C):
        for j in range(C):
            c = i * (C + 1) + j
            complexes.append([c, c + 1, c + (C + 1) + 1, c + (C + 1)])

    complexes = np.array(complexes).reshape((C * C, 4))

    print('complexes:', complexes.shape)
    print('points:', points.shape)
    print('corners:', corners.shape)

    print(complexes)

    return complexes, points, corners
    # complexes = np.array([[0, 1, 2, 3]])
    # points = []
    #
    # # [0, 1] dimensions
    # for i in range(sample_rate):
    #     for j in range(sample_rate):
    #         u, v = (i / sample_rate), (j / sample_rate)
    #         v = np.array([u, u + v, v]) / sample_rate
    #         points.append(v)
    #
    # points = np.array(points)
    # corners = [ points[0], points[sample_rate - 1], points[-1], points[-sample_rate] ]
    # return complexes, points, np.array(corners)

def curved(sample_rate=16):
    complexes = np.array([[0, 1, 2, 3]])
    points = []

    # [0, 1] dimensions
    for i in range(sample_rate):
        for j in range(sample_rate):
            u, v = (i / sample_rate), (j / sample_rate)
            Y = (u + v) ** 2
            v = np.array([u, 0.25 * Y, v])
            points.append(v)

    points = np.array(points)
    corners = [ points[0], points[sample_rate - 1], points[-1], points[-sample_rate] ]
    return complexes, points, np.array(corners)

def perlin(sample_rate=16):
    from perlin_noise import PerlinNoise

    noise = PerlinNoise(octaves=16, seed=1)
    complexes = np.array([[0, 1, 2, 3]])
    points = []

    # [0, 1] dimensions
    for i in range(sample_rate):
        for j in range(sample_rate):
            u, v = (i / sample_rate), (j / sample_rate)
            Y = noise([u, v])
            v = np.array([u, 0.25 * Y, v])
            points.append(v)

    points = np.array(points)
    corners = [ points[0], points[sample_rate - 1], points[-1], points[-sample_rate] ]
    return complexes, points, np.array(corners)

# TODO: brick texture or so
def textured(image, amp=0.1):
    from PIL import Image

    # img = Image.open('images/lizard.png')
    def ftn(sample_rate=16):
        img = Image.open(image)
        complexes = np.array([[0, 1, 2, 3]])
        points = []

        # [0, 1] dimensions
        for i in range(sample_rate):
            for j in range(sample_rate):
                u, v = (i / sample_rate), (j / sample_rate)
                d = img.getpixel((u * img.width, v * img.height))[0]/255.0
                v = np.array([u, amp * d, v])
                points.append(v)

        points = np.array(points)
        corners = [ points[0], points[sample_rate - 1], points[-1], points[-sample_rate] ]
        return complexes, points, np.array(corners)

    return ftn

# TODO: custom 3D viewer...
def quadify(C, sample_rate=16):
    quads = []
    for c in range(C.shape[0]):
        offset = c * sample_rate * sample_rate
        for i in range(sample_rate - 1):
            for j in range(sample_rate - 1):
                a = offset + i * sample_rate + j
                c = offset + (i + 1) * sample_rate + j
                b, d = a + 1, c + 1
                quads.append([b, d, c, a])

    return np.array(quads)

def indices(C, sample_rate=16):
    triangles = []
    for c in range(C.shape[0]):
        offset = c * sample_rate * sample_rate
        for i in range(sample_rate - 1):
            for j in range(sample_rate - 1):
                a = offset + i * sample_rate + j
                c = offset + (i + 1) * sample_rate + j
                b, d = a + 1, c + 1
                triangles.append([a, b, c])
                triangles.append([b, d, c])

    return np.array(triangles)

# TODO: test different texture maps

experiments = {
    'flat':   flat,
    'linear': linear,
    # 'curved':   curved,
    # 'perlin':   perlin,
    # 'tiles':    textured('images/tiles.png'),
    # 'fabric':   textured('images/fabric.png'),
    # 'wicker':   textured('../images/wicker.png'),
    # 'column':   textured('../images/column.png'),
}

kernels = {
        'linear' : clerp(lerps['linear']),
        'floor'  : clerp(lerps['floor']),
        'sin'    : clerp(lerps['sin']),
        'sincos' : clerp(lerps['sincos']),
}

# TODO: also the lerp functions; separate file tho...
# morlet and onion seems to get rid of the ripple artifacts
# feature encoding seems to be better (at least in single patches)

ps.init()

# Graphing
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi']  = 600

N = len(kernels)
fig_viz, axs_viz = plt.subplots(len(experiments), N, figsize=(30, 20), gridspec_kw={ 'wspace': 0.05, 'hspace': 0.01 })
fig_nrm, axs_nrm = plt.subplots(len(experiments), 2 * N + 1, figsize=(22, 7), gridspec_kw={ 'wspace': 0.05, 'hspace': 0.01 })

from torchmetrics.image import PeakSignalNoiseRatio
psnr = PeakSignalNoiseRatio().cuda()

sample_rate       = 4 # 256 # TODO: try 1024...
complexes_size    = 2
super_sample_rate = complexes_size * sample_rate

for i, (name, experiment) in enumerate(experiments.items()):
    # Generate reference
    complexes, target, corners = experiment(sample_rate, C=complexes_size)

    # Precompute indices
    triangles = indices(complexes, sample_rate)
    tch_triangles = torch.from_numpy(triangles).int().cuda()

    # Precompute normals
    tch_target = torch.from_numpy(target).float().cuda()

    normals_true = compute_face_normals(tch_target, tch_triangles)
    normals_true = compute_vertex_normals(tch_target, tch_triangles, normals_true)
    normals_true_viz = normals_true.detach().cpu().numpy()
    normals_true_viz = normals_true_viz.reshape(super_sample_rate, super_sample_rate, 3)
    normals_true_viz = normals_true_viz[:, :, [0, 2, 1]]

    axs_nrm[i, 0].imshow(normals_true_viz * 0.5 + 0.5)
    axs_nrm[i, 0].set_ylabel(f'\\textsc{{{name}}}')
    axs_nrm[i, 0].grid(False)
    axs_nrm[i, 0].set_xticks([])
    axs_nrm[i, 0].set_yticks([])

    if i == 0:
        axs_nrm[i, 0].set_title(r'\textsc{Normals}')

    vizes = []
    viz_max = 0.0

    nrm_vizes = []
    nrm_viz_max = 0.0

    for j, (kernel_name, K) in enumerate(kernels.items()):
        # Load the model and parameters
        torch.manual_seed(0)
        m = MLP_Positional_Onion_Encoding().cuda()
        # s = sampler(kernel=K)

        tch_complexes = torch.from_numpy(complexes).int().cuda()
        tch_corners   = torch.from_numpy(corners).float().cuda()
        tch_encodings = torch.randn((corners.shape[0], POINT_ENCODING_SIZE), requires_grad=True, dtype=torch.float32, device='cuda')

        # Train the model
        optimizer = torch.optim.Adam(list(m.parameters()) + [ tch_encodings ], lr=1e-3)

        history = {}
        for _ in trange(1_000):
            LP, LE = sample(tch_complexes, tch_corners, tch_encodings, sample_rate)
            V = m(points=LP, features=LE)

            normals_pred = compute_face_normals(V, tch_triangles)
            normals_pred = compute_vertex_normals(V, tch_triangles, normals_pred)

            vertex_loss = torch.mean(torch.norm(V - tch_target, dim=1))
            normal_loss = 1e-1 * torch.mean(torch.norm(normals_true - normals_pred, dim=1))

            loss = vertex_loss # + normal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final evaluation and visualization
        LP, LE = sample(tch_complexes, tch_corners, tch_encodings, sample_rate)
        V = m(points=LP, features=LE)

        quads = quadify(complexes, sample_rate)
        off_target = target + np.array([1.5 * i, 0, 0])
        # ref_mesh = ps.register_surface_mesh(name, off_target, quads, color=(0.5, 1.0, 0.5))
        ref_mesh = ps.register_surface_mesh(name, off_target, tch_triangles.cpu().numpy(), color=(0.5, 1.0, 0.5))
        ref_mesh.add_scalar_quantity('Y', target[:, 1], defined_on='vertices')

        off_model = V.detach().cpu().numpy() + np.array([1.5 * i, 0, 1.5 * (j + 1)])
        # nsc_mesh = ps.register_surface_mesh(name + ':' + kernel_name, off_model, quads, color=(0.5, 0.5, 1.0))
        nsc_mesh = ps.register_surface_mesh(name + ':' + kernel_name, off_model, tch_triangles.cpu().numpy(), color=(0.5, 0.5, 1.0))

        # Final error as an image
        error = (V - tch_target).norm(dim=1)
        print('error shape: {}', error.shape)
        viz = axs_viz[i, j].imshow(error.reshape(complexes_size * sample_rate, complexes_size * sample_rate).detach().cpu().numpy())
        axs_viz[i, j].grid(False)
        axs_viz[i, j].set_xticks([])
        axs_viz[i, j].set_yticks([])
        vizes.append(viz)
        viz_max = max(viz_max, error.max().item())

        print(f'{name:<15} {kernel_name:<15} |    u = {error.mean().item():.5f}    |    sigma = {error.std().item():.3f}    |    max = {error.max().item():.3f}')

        # Display statistics below the image
        axs_viz[i, j].set_xlabel(f'$\\mu = {error.mean().item():.3f},\\;\\;\\sigma = {error.std().item():.3f}$')

        # normals_pred = compute_face_normals(V.detach(), tch_triangles)
        # normals_pred = compute_vertex_normals(V.detach(), tch_triangles, normals_pred)
        # normals_pred = normals_pred.detach().cpu().numpy()
        #
        # normals_pred = normals_pred.reshape(sample_rate, sample_rate, 3)
        #
        # # Swap the Y and Z axis
        # normals_pred = normals_pred[:, :, [0, 2, 1]]
        #
        # axs_nrm[i, j + 1].imshow(normals_pred * 0.5 + 0.5)
        # axs_nrm[i, j + 1].grid(False)
        # axs_nrm[i, j + 1].set_xticks([])
        # axs_nrm[i, j + 1].set_yticks([])
        #
        # psnr_value = psnr(torch.from_numpy(normals_pred * 0.5 + 0.5).cuda(), torch.from_numpy(normals_true_viz * 0.5 + 0.5).cuda())
        # l2_value = np.linalg.norm(normals_pred - normals_true_viz, axis=2).mean()
        #
        # axs_nrm[i, j + 1].set_xlabel(f'${psnr_value:.2f}$ / ${l2_value:.2f}$')
        #
        # print(f'{name:<15} {kernel_name:<15} |   PSNR = {psnr_value:.3f}   |    L2 = {l2_value:.3f}')
        #
        # normal_diff = (normals_true_viz - normals_pred)
        # normal_diff = np.linalg.norm(normal_diff, axis=2)
        # nrm_viz = axs_nrm[i, N + j + 1].imshow(normal_diff)
        # axs_nrm[i, N + j + 1].grid(False)
        # axs_nrm[i, N + j + 1].set_xticks([])
        # axs_nrm[i, N + j + 1].set_yticks([])
        #
        # nrm_vizes.append(nrm_viz)
        # nrm_viz_max = max(nrm_viz_max, normal_diff.max())
        #
        # if i == 0:
        #     axs_viz[0, j].set_title(f'\\textsc{{{kernel_name}}}')
        #     axs_nrm[0, j + 1].set_title(f'\\textsc{{{kernel_name}}}')
        #     axs_nrm[0, N + j + 1].set_title(f'\\textsc{{{kernel_name}}}')
        #
        # if j == 0:
        #     axs_viz[i, 0].set_ylabel(f'\\textsc{{{name}}}')

    for k in range(N):
        vizes[k].set_clim(0, viz_max)
        vizes[k].set_cmap('inferno')

        # nrm_vizes[k].set_clim(0, nrm_viz_max)
        # nrm_vizes[k].set_cmap('inferno')

    # Colorbar only on the last column
    fig_viz.colorbar(vizes[N - 1], ax=axs_viz[i, :], shrink=0.7, pad=0.01, format='%.2f')
    # fig_nrm.colorbar(nrm_vizes[N - 1], ax=axs_nrm[i, :], shrink=0.7, pad=0.01, format='%.2f')

ps.show()

fig_viz.savefig('kernels-pos.png', bbox_inches='tight')
# fig_nrm.savefig('kernels-nrm.png', bbox_inches='tight')
