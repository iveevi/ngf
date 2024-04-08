import os
import json
import tqdm
import torch
import logging
import ngfutil
import argparse
import pymeshlab
import trimesh

from util import *
from ngf import NGF
from render import Renderer


def uniform_laplacian(Q: torch.Tensor, N: int, lambda_: float = 10.0) -> torch.Tensor:
    import itertools

    graph = [[] for _ in range(N)]
    for q in Q:
        l = q.cpu().numpy().tolist()
        for p in itertools.combinations(l, 2):
            graph[p[0]].append(p[1])
            graph[p[1]].append(p[0])

    # print(graph)

    values = []
    ix, iy = [], []
    for i, g in enumerate(graph):
        values.append(lambda_ - len(g))
        ix.append(i)
        iy.append(i)

        for k in g:
            values.append(1)
            ix.append(i)
            iy.append(k)

    return torch.sparse_coo_tensor([ix, iy], values, size=(N, N)).cuda()


class Trainer:
    def __init__(self, mesh: str, lod: int, features: int = 20):
        # Properties
        self.path = os.path.abspath(mesh)
        self.cameras = 200
        self.batch = 10
        self.losses = {}

        logging.info('Launching training process with configuration:')
        logging.info(f'    Reference mesh: {self.path}')
        logging.info(f'    Camera count:   {self.cameras}')
        logging.info(f'    Batch size:     {self.batch}')

        self.exporter = Exporter(mesh, lod)

        self.target, normalizer = load_mesh(mesh)
        logging.info(f'Loaded reference mesh {mesh}')

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(mesh)
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2 * lod)
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_tri_to_quad_by_smart_triangle_pairing()
        ms.save_current_mesh(self.exporter.partitioned())
        logging.info(f'Quadrangulated mesh into {self.exporter.partitioned()}')

        self.renderer = Renderer()
        logging.info('Constructed renderer for optimization')

        self.views: torch.Tensor = None
        self.reference_views: torch.Tensor = None

        self.ngf = NGF.from_base(self.exporter.partitioned(), normalizer, features)

    def precompute_reference_views(self):
        vertices = self.target.vertices
        vertices = vertices[self.target.faces].reshape(-1, 3)
        faces = torch.arange(vertices.shape[0])
        faces = faces.int().cuda().reshape(-1, 3)
        normals = vertex_normals(vertices, faces)

        cache = []
        for view in tqdm.tqdm(self.views, ncols=50, leave=False):
            reference_view = self.renderer.render(vertices, normals, faces, view.unsqueeze(0))
            # reference_view = self.renderer.shaded(vertices, normals, faces, view.unsqueeze(0))
            cache.append(reference_view)

        return list(torch.cat(cache).split(self.batch))

    def initialize_ngf(self) -> None:
        with torch.no_grad():
            base = self.ngf.base(4)

        opt = torch.optim.Adam(self.ngf.parameters(), 1e-3)
        for _ in tqdm.trange(1_000, ncols=50, leave=False):
            uvs = self.ngf.sample_uniform(4)
            vertices = self.ngf.eval(*uvs)

            loss = (vertices - base).abs().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        logging.info('Neural geometry field initialization phase done')

    def optimize_resolution(self, optimizer: torch.optim.Optimizer, rate: int) -> dict[str, list[float]]:
        import numpy as np

        losses = {
            'render': [],
            'laplacian': []
        }

        base = self.ngf.base(rate).detach()
        cmap = make_cmap(self.ngf.complexes, self.ngf.points.detach(), base, rate)
        remap = ngfutil.generate_remapper(self.ngf.complexes.cpu(), cmap, base.shape[0], rate)
        quads = torch.from_numpy(quadify(self.ngf.complexes.shape[0], rate)).int()
        graph = ngfutil.Graph(remap.remap(quads), base.shape[0])

        batched_views = list(self.views.split(self.batch))
        indices = np.arange(len(batched_views))

        for _ in tqdm.trange(200, ncols=50, leave=False):
            batch_losses = {
                'render': [],
                'laplacian': []
            }

            uvs = self.ngf.sampler(rate)
            uniform_uvs = self.ngf.sample_uniform(rate)

            np.random.shuffle(indices)
            for i in indices:
                batch_reference_views = self.reference_views[i]
                batch_views = batched_views[i]

                vertices = self.ngf.eval(*uvs)
                uniform_vertices = self.ngf.eval(*uniform_uvs)

                faces = ngfutil.triangulate_shorted(vertices, self.ngf.complexes.shape[0], rate)
                faces = remap.remap_device(faces)
                normals = vertex_normals(vertices, faces)

                smoothed_vertices = graph.smooth(uniform_vertices, 1.0)
                laplacian_loss = (uniform_vertices - smoothed_vertices).abs().mean()

                batch_source_views = self.renderer.render(vertices, normals, faces, batch_views)

                render_loss = (batch_reference_views.cuda() - batch_source_views).abs().mean()
                loss = render_loss + laplacian_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses['render'].append(render_loss.item())
                batch_losses['laplacian'].append(laplacian_loss.item())

            losses['render'].append(np.mean(batch_losses['render']))
            losses['laplacian'].append(np.mean(batch_losses['laplacian']))

        logging.info(f'Optimized neural geometry field at resolution ({rate} x {rate})')

        return losses

    # def optimize_resolution_diffused(self, optimizer: torch.optim.Optimizer, rate: int) -> dict[str, list[float]]:
    #     import numpy as np
    #
    #     losses = {
    #         'render': [],
    #         'laplacian': []
    #     }
    #
    #     base = self.ngf.base(rate).detach()
    #     cmap = make_cmap(self.ngf.complexes, self.ngf.points.detach(), base, rate)
    #     remap = ngfutil.generate_remapper(self.ngf.complexes.cpu(), cmap, base.shape[0], rate)
    #     quads = torch.from_numpy(quadify(self.ngf.complexes.shape[0], rate)).int()
    #     graph = ngfutil.Graph(remap.remap(quads), base.shape[0])
    #
    #     dmatrix = uniform_laplacian(quads, base.shape[0])
    #
    #     batched_views = list(self.views.split(self.batch))
    #     indices = np.arange(len(batched_views))
    #
    #     for _ in tqdm.trange(200, ncols=50, leave=False):
    #         batch_losses = {
    #             'render': [],
    #             'laplacian': []
    #         }
    #
    #         uvs = self.ngf.sampler(rate)
    #         uniform_uvs = self.ngf.sample_uniform(rate)
    #
    #         np.random.shuffle(indices)
    #         for i in indices:
    #             batch_reference_views = self.reference_views[i]
    #             batch_views = batched_views[i]
    #
    #             with torch.no_grad():
    #                 vertices = self.ngf.eval(*uvs)
    #                 vertices.requires_grad = True
    #
    #             uniform_vertices = self.ngf.eval(*uniform_uvs)
    #
    #             faces = ngfutil.triangulate_shorted(vertices, self.ngf.complexes.shape[0], rate)
    #             faces = remap.remap_device(faces)
    #             normals = vertex_normals(vertices, faces)
    #
    #             # smoothed_vertices = graph.smooth(uniform_vertices, 1.0)
    #             # laplacian_loss = (uniform_vertices - smoothed_vertices).abs().mean()
    #
    #             batch_source_views = self.renderer.render(vertices, normals, faces, batch_views)
    #
    #             render_loss = (batch_reference_views.cuda() - batch_source_views).abs().mean()
    #             loss = render_loss #+ laplacian_loss
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #
    #             with torch.no_grad():
    #                 vertices.grad = dmatrix @ vertices.grad
    #
    #             proxy_vertices = self.ngf.eval(*uvs)
    #             proxy_vertices.backward(vertices.grad)
    #             optimizer.step()
    #
    #             batch_losses['render'].append(render_loss.item())
    #             # batch_losses['laplacian'].append(laplacian_loss.item())
    #             batch_losses['laplacian'].append(0)
    #
    #         losses['render'].append(np.mean(batch_losses['render']))
    #         losses['laplacian'].append(np.mean(batch_losses['laplacian']))
    #
    #     logging.info(f'Optimized neural geometry field at resolution ({rate} x {rate})')
    #
    #     return losses

    def optimize_resolution_indirectly(self, optimizer: torch.optim.Optimizer, rate: int) -> dict[str, list[float]]:
        import numpy as np
        import polyscope as ps

        from largesteps.geometry import compute_matrix
        from largesteps.optimize import AdamUniform
        from largesteps.parameterize import from_differential, to_differential

        losses = {
            'render': [],
            'laplacian': []
        }

        # TODO: how to do jittering?
        base = self.ngf.base(rate).detach()
        cmap = make_cmap(self.ngf.complexes, self.ngf.points.detach(), base, rate)
        remap = ngfutil.generate_remapper(self.ngf.complexes.cpu(), cmap, base.shape[0], rate)

        batched_views = list(self.views.split(self.batch))
        indices = np.arange(len(batched_views))

        lambda_, lr = {
            4: (10, 1e-2),
            8: (20, 5e-3),
            16: (30, 1e-3)
        }[rate]

        with torch.no_grad():
            uvs = self.ngf.sample_uniform(rate)
            vertices = self.ngf.eval(*uvs)

            faces = ngfutil.triangulate_shorted(vertices, self.ngf.complexes.shape[0], rate)
            faces = remap.remap_device(faces)

            M = compute_matrix(vertices, faces, lambda_=lambda_)
            U = to_differential(M, vertices)
            U.requires_grad = True

        opt = AdamUniform([U], lr)

        ps.init()
        ps.register_surface_mesh('initial', vertices.detach().cpu().numpy(), faces.cpu().numpy())

        for _ in tqdm.trange(200, ncols=50, leave=False):
            np.random.shuffle(indices)

            batch = []
            for i in indices:
                batch_reference_views = self.reference_views[i]
                batch_views = batched_views[i]

                vertices = from_differential(M, U, 'Cholesky')
                normals = vertex_normals(vertices, faces)

                batch_source_views = self.renderer.render(vertices, normals, faces, batch_views)
                # batch_source_views = self.renderer.shaded(vertices, normals, faces, batch_views)

                render_loss = (batch_reference_views.cuda() - batch_source_views).abs().mean()

                opt.zero_grad()
                render_loss.backward()
                opt.step()

                # losses['render'].append(render_loss.item())
                batch.append(render_loss.item())

            losses['render'].append(np.mean(batch))

        true_vertices = remap.scatter_device(vertices.detach())

        optimizer = torch.optim.Adam(self.ngf.parameters(), 1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)

        for _ in tqdm.trange(10000, ncols=50, leave=False):
            vertices = self.ngf.eval(*uvs)
            loss = (true_vertices - vertices).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses['laplacian'].append(loss.item())

        ps.register_surface_mesh('final', true_vertices.cpu().numpy(), faces.cpu().numpy())
        ps.register_surface_mesh('learned', vertices.detach().cpu().numpy(), faces.cpu().numpy())

        ps.show()

        logging.info(f'Optimized neural geometry field at resolution ({rate} x {rate})')

        return losses

    def run(self) -> None:
        # self.initialize_ngf()

        self.losses = {
            'render': [],
            'laplacian': []
        }

        self.views = arrange_views(self.target, self.cameras)[0]
        logging.info(f'Generated {self.cameras} views for reference mesh')

        self.reference_views = self.precompute_reference_views()
        logging.info('Cached reference views')

        for rate in [4, 8, 16]:
            opt = torch.optim.Adam(self.ngf.parameters(), 1e-3)
            rate_losses = self.optimize_resolution(opt, rate)
            # rate_losses = self.optimize_resolution_diffused(opt, rate)
            # rate_losses = self.optimize_resolution_indirectly(opt, rate)
            self.losses['render'] += rate_losses['render']
            self.losses['laplacian'] += rate_losses['laplacian']

        logging.info('Finished training neural geometry field')

    def export(self) -> None:
        # Final export
        self.ngf.save(self.exporter.pytorch())

        logging.info('Exporting neural geometry field as PyTorch (PT)')

        with open(self.exporter.binary(), 'wb') as file:
            file.write(self.ngf.stream())

        logging.info('Exporting neural geometry field as binary')

        # Plot results
        fig, axs = plt.subplots(1, 2, layout='constrained')

        axs[0].plot(self.losses['render'], label='Render')
        axs[0].legend()
        axs[0].set_yscale('log')

        axs[1].plot(self.losses['laplacian'], label='Laplacian')
        axs[1].legend()
        axs[1].set_yscale('log')

        plt.savefig(self.exporter.plot())

        logging.info('Loss history exported')

        # Export mesh
        uvs = self.ngf.sample_uniform(16)
        vertices = self.ngf.eval(*uvs).detach()
        base = self.ngf.base(16).detach()
        cmap = make_cmap(self.ngf.complexes, self.ngf.points.detach(), base, 16)
        remap = ngfutil.generate_remapper(self.ngf.complexes.cpu(), cmap, base.shape[0], 16)
        faces = ngfutil.triangulate_shorted(vertices, self.ngf.complexes.shape[0], 16)
        faces = remap.remap_device(faces)

        mesh = trimesh.Trimesh(vertices=vertices.cpu(), faces=faces.cpu())
        mesh.export(self.exporter.mesh())

        # Write the metadate
        meta = {
            'reference': self.path,
            'torched': os.path.abspath(self.exporter.pytorch()),
            'binaries': os.path.abspath(self.exporter.binary()),
            'stl': os.path.abspath(self.exporter.mesh())
        }

        with open(self.exporter.metadata(), 'w') as file:
            json.dump(meta, file)

    def display(self, rate=16):
        import polyscope as ps

        ps.init()
        ps.register_surface_mesh('Reference',
                                 self.target.vertices.cpu().numpy(),
                                 self.target.faces.cpu().numpy())

        with torch.no_grad():
            base = self.ngf.base(rate).float()
            uvs = self.ngf.sample_uniform(rate)
            vertices = self.ngf.eval(*uvs).float()

        cmap = make_cmap(self.ngf.complexes, self.ngf.points.detach(), base, rate)
        remap = ngfutil.generate_remapper(self.ngf.complexes.cpu(), cmap, base.shape[0], rate)
        faces = ngfutil.triangulate_shorted(vertices, self.ngf.complexes.shape[0], rate)
        faces = remap.remap_device(faces)

        ps.register_surface_mesh('NGF', vertices.cpu().numpy(), faces.cpu().numpy())
        ps.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str)
    parser.add_argument('--lod', type=int, default=2000)
    parser.add_argument('--display', type=bool, default=True)

    args = parser.parse_args()

    trainer = Trainer(args.mesh, args.lod)
    trainer.run()
    trainer.export()

    if args.display:
        trainer.display()
