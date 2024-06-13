import sys
import tqdm
import numpy as np
import pprint
import argparse

from torchmetrics.image import PeakSignalNoiseRatio
from kaolin.metrics.pointcloud import chamfer_distance

from util import *
from ngf import NGF
from render import Renderer

if __name__ == '__main__':
    VIEWLUT = {}

    parser = argparse.ArgumentParser()
    # parser.add_argument('--ref', type=int, default=0)
    parser.add_argument('--meshes', type=str, nargs='+')
    parser.add_argument('--cameras', type=int, default=200)
    parser.add_argument('--key', type=str, choices=['lod', 'rate'])

    args = parser.parse_args(sys.argv[1:])

    meshes = []
    keys = {}

    print('MESHES')
    for i, mesh in enumerate(args.meshes):
        print(f'[{i:2d}]', mesh, end='')

        normalizer = lambda x: x
        # if input('normalize? [y/n] ') == 'y':
        if 'lod' not in mesh or 'qslim' in mesh or 'nvdiffmodeling' in mesh:
            print(' -- [normalized]')
            normalizer = None
        else:
            print()

        if 'lod' in mesh:
            basename = os.path.basename(mesh)
            basename = basename.split('.')[0]
            splits = basename.split('-')
            keys[mesh] = {}
            for s in splits:
                if s.startswith('lod'):
                    keys[mesh]['lod'] = int(s[3:])
                if s.startswith('f'):
                    keys[mesh]['feature'] = int(s[1:])
                if s.startswith('r'):
                    keys[mesh]['rate'] = int(s[1:])

        mesh, normalizer = load_mesh(mesh, normalizer)
        meshes.append(mesh)

    import polyscope as ps
    ps.init()
    for i, mesh in enumerate(meshes):
        ps.register_surface_mesh(f'mesh{i}', mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy())
    ps.show()

    refi = int(input('\nWhich mesh is the reference? '))

    reference = meshes[refi]
    print('\nREFERENCE', args.meshes[refi])
        
    views = arrange_views(reference, args.cameras)[0]
    print('\nGENERATED', args.cameras, 'CAMERAS')

    renderer = Renderer(1920, 1080)
    psnr = PeakSignalNoiseRatio().cuda()

    shaded_metrics = {}
    normal_metrics = {}
    chamfer_metrics = {}

    for view in tqdm.tqdm(views, ncols=50, leave=False):
        # TODO: do not separate
        normals = vertex_normals(reference.vertices, reference.faces)
        shaded = renderer.shaded(reference.vertices, normals, reference.faces, view[None])[0]
        normal = renderer.interpolate(reference.vertices, normals, reference.faces, view[None])[0]
        assert not shaded.isnan().any()

        # shaded = renderer.shaded(*separate(reference.vertices, reference.faces), view[None])[0]
        # normal = renderer.interpolate(*separate(reference.vertices, reference.faces), view[None])[0]

        # plt.imshow(shaded[0].cpu()/1000)
        # plt.axis('off')
        # plt.show()
        #
        # plt.imshow(normal[0].cpu() * 0.5 + 0.5)
        # plt.axis('off')
        # plt.show()

        # fig, axs = plt.subplots(2, 1 + len(meshes))
        # for ax in axs.flatten():
        #     ax.axis('off')
        #
        # axs[0][0].imshow(normal.cpu() * 0.5 + 0.5)

        for i, mesh in enumerate(meshes):
            if i == refi:
                continue

            # mesh_shaded = renderer.shaded(*separate(mesh.vertices, mesh.faces), view[None])[0]
            # mesh_normal = renderer.interpolate(*separate(mesh.vertices, mesh.faces), view[None])[0]
            mesh_normals = vertex_normals(mesh.vertices, mesh.faces)
            mesh_shaded = renderer.shaded(mesh.vertices, mesh_normals, mesh.faces, view[None])[0]
            mesh_normal = renderer.interpolate(mesh.vertices, mesh_normals, mesh.faces, view[None])[0]
            assert not mesh_shaded.isnan().any()

            # axs[0][1 + i].set_title(args.meshes[i])
            # axs[0][1 + i].imshow(mesh_normal.cpu() * 0.5 + 0.5)
            # diffmag = (normal - mesh_normal).square().mean(dim=-1)
            # c = axs[1][1 + i].imshow(diffmag.cpu())
            # plt.colorbar(c)

            error_shaded = (shaded - mesh_shaded).square().mean().item()
            error_normal = (normal - mesh_normal).square().mean().item()
            
            shaded_metrics.setdefault(i, []).append(error_shaded)
            normal_metrics.setdefault(i, []).append(error_normal)

        # plt.show()
        
    for i, mesh in enumerate(meshes):
        if i == refi:
            continue

        chamfer = chamfer_distance(reference.vertices[None, ...], mesh.vertices[None, ...])
        print('chamfer', chamfer.item())

        chamfer_metrics[i] = chamfer.item()

    for i, mesh in enumerate(args.meshes):
        if i == refi:
            continue

        shaded_metrics[mesh] = np.mean(shaded_metrics[i])
        normal_metrics[mesh] = np.mean(normal_metrics[i])
        chamfer_metrics[mesh] = chamfer_metrics[i]

        del shaded_metrics[i]
        del normal_metrics[i]
        del chamfer_metrics[i]

    print('\nGROUND TRUTH VIEWS EVALUATED (%d x %d)' % (1920, 1080))

    for i, mesh in enumerate(args.meshes):
        if i == refi:
            continue

        print('\nMETRICS FOR', mesh)
        print(' * SHADED   %.3e' % shaded_metrics[mesh])
        print(' * NORMAL   %.3e' % normal_metrics[mesh])
        print(' * CHAMFER  %.3e' % chamfer_metrics[mesh])

    shaded_points = []
    normal_points = []
    chamfer_points = []

    for i, mesh in enumerate(args.meshes):
        if i == refi:
            continue

        k = keys[mesh][args.key]
        shaded_points.append((k, shaded_metrics[mesh]))
        normal_points.append((k, normal_metrics[mesh]))
        chamfer_points.append((k, chamfer_metrics[mesh]))

    shaded_points = sorted(shaded_points, key=lambda v: v[0])
    normal_points = sorted(normal_points, key=lambda v: v[0])
    chamfer_points = sorted(chamfer_points, key=lambda v: v[0])

    print('\nPLOTS')
    
    print('\nSHADED')
    for p in shaded_points:
        print('  (%4d, %.3e)' % p)
    
    print('\nNORMAL')
    for p in normal_points:
        print('  (%4d, %.3e)' % p)
    
    print('\nCHAMFER')
    for p in chamfer_points:
        print('  (%4d, %.3e)' % p)
