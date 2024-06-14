import os
import json
import tqdm
import torch
import logging
import ngfutil
import argparse
import pymeshlab
import trimesh
import multiprocessing

from util import *
from ngf import NGF
from render import Renderer


class Trainer:
    @staticmethod
    def quadrangulate_surface(mesh: str, count: int, destination: str) -> None:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(mesh)
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=count, qualitythr=1.0)
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_tri_to_quad_by_smart_triangle_pairing()
        ms.save_current_mesh(destination)
        logging.info(f'Quadrangulated mesh into {destination}')

    def __init__(self, mesh: str, lod: int, features: int, batch: int):
        # Properties
        self.path = os.path.abspath(mesh)
        self.cameras = 200
        self.batch = batch
        self.losses = {}

        logging.info('Launching training process with configuration:')
        logging.info(f'    Reference mesh: {self.path}')
        logging.info(f'    Camera count:   {self.cameras}')
        logging.info(f'    Batch size:     {self.batch}')

        self.exporter = Exporter(mesh, lod, features)

        self.target, normalizer = load_mesh(mesh)
        logging.info(f'Loaded reference mesh {mesh}')

        qargs = (mesh, 2 * lod, self.exporter.partitioned())
        proc = multiprocessing.Process(target=Trainer.quadrangulate_surface, args=qargs)
        proc.start()

        # Wait for minute before termimating
        proc.join(60)
        if proc.is_alive():
            logging.error('Quadrangulation running overtime')
            proc.terminate()
            exit()

        self.renderer = Renderer()
        logging.info('Constructed renderer for optimization')

        self.views = None
        self.reference_views = None

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
        length = average_edge_length(base, quads)

        for _ in tqdm.trange(100, ncols=50, leave=False):
            batch_losses = {
                'render': [],
                'laplacian': []
            }

            uvs = self.ngf.sampler(rate)
            uniform_uvs = self.ngf.sample_uniform(rate)

            for batch_views, ref_views in zip(batched_views, self.reference_views):
                vertices = self.ngf.eval(*uvs)
                uniform_vertices = self.ngf.eval(*uniform_uvs)

                faces = ngfutil.triangulate_shorted(vertices, self.ngf.complexes.shape[0], rate)
                faces = remap.remap_device(faces)

                # TODO: flatter normals
                vertices, normals, faces = separate(vertices, faces)
                # normals = vertex_normals(vertices, faces)

                smoothed_vertices = graph.smooth(uniform_vertices, 1.0)
                smoothed_vertices = remap.scatter_device(smoothed_vertices)
                laplacian_loss = (uniform_vertices - smoothed_vertices).abs().mean()

                batch_source_views = self.renderer.render(vertices, normals, faces, batch_views)

                render_loss = (ref_views.cuda() - batch_source_views).abs().mean()
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

        for rate in [4, 8, 12, 16]:
            opt = torch.optim.Adam(self.ngf.parameters(), 1e-3)
            rate_losses = self.optimize_resolution(opt, rate)
            self.losses['render'] += rate_losses['render']
            self.losses['laplacian'] += rate_losses['laplacian']
            # self.display(rate)

        logging.info('Finished training neural geometry field')

    def export(self) -> None:
        # Final export
        self.ngf.save(self.exporter.pytorch())

        logging.info('Exporting neural geometry field as PyTorch (PT)')

        with open(self.exporter.binary(), 'wb') as file:
            file.write(self.ngf.stream())

        logging.info('Exporting neural geometry field as binary')

        # Plot results
        _, axs = plt.subplots(1, 2, layout='constrained')

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
    parser.add_argument('--mesh', type=str, help='Target mesh')
    parser.add_argument('--lod', type=int, default=2000, help='Number of patches to partition')
    parser.add_argument('--features', type=int, default=20, help='Feature vector size')
    parser.add_argument('--display', type=bool, default=True, help='Display the result after training')
    parser.add_argument('--batch', type=int, default=10, help='Batch size for training')
    parser.add_argument('--fixed-seed', action='store_true', default=False, help='Fixed random seed (for debugging)')

    args = parser.parse_args()
    print(args)

    if args.fixed_seed:
        torch.manual_seed(0)

    trainer = Trainer(args.mesh, args.lod, args.features, args.batch)
    trainer.run()
    trainer.export()

    if args.display:
        trainer.display()
