import argparse
import logging
import torch
import tqdm
import optext

from util import *
from ngf import NGF
from render import Renderer


class Trainer:
    def __init__(self, mesh: str, base: str, features: int = 10):
        # Properties
        # TODO: dump in the log
        self.batch = 20

        self.target, normalizer = load_mesh(mesh)
        logging.info(f'Loaded reference mesh {mesh}')

        self.renderer = Renderer()
        logging.info('Constructed renderer for optimization')

        self.views = arrange_views(self.target, 200)
        logging.info('Generated views for reference mesh')

        self.reference_views = self.reference_views()
        logging.info('Cached reference views')

        self.ngf = NGF.from_base(base, normalizer, features)
        logging.info('Instantiated neural geometry field')

    def reference_views(self):
        vertices = self.target.vertices
        vertices = vertices[self.target.faces].reshape(-1, 3)

        faces = torch.arange(vertices.shape[0])
        faces = faces.int().cuda().reshape(-1, 3)

        normals = vertex_normals(vertices, faces)

        cache = []
        for batch_views in self.views.split(self.batch):
            # ref_data = renderer.render_attributes(VT, NT, FT, view_mats)
            reference_views = self.renderer.shlighting(vertices, normals, faces, batch_views)
            cache.append(reference_views.cpu())

        return cache

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

        logging.info('Initialized neural geometry field with base')

    def optimize_resolution(self, optimizer: torch.optim.Optimizer, rate: int) -> list[float]:
        losses = []

        base = self.ngf.base(rate).detach()
        cmap = make_cmap(self.ngf.complexes, self.ngf.points.detach(), base, rate)
        remap = optext.generate_remapper(self.ngf.complexes.cpu(), cmap, base.shape[0], rate)

        batched_views = self.views.split(self.batch)
        for _ in tqdm.trange(100, ncols=50, leave=False):
            batch_losses = []
            for batch_reference_views, batch_views in zip(self.reference_views, batched_views):
                uvs = self.ngf.sampler(rate)
                vertices = self.ngf.eval(*uvs)

                faces = optext.triangulate_shorted(vertices, self.ngf.complexes.shape[0], rate)
                faces = remap.remap_device(faces)
                normals = vertex_normals(vertices, faces)

                batch_source_views = self.renderer.shlighting(vertices, normals, faces, batch_views)

                loss = (batch_reference_views.cuda() - batch_source_views).abs().mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            losses.append(np.mean(batch_losses))

        logging.info(f'Optimized neural geometry field at {rate}x{rate}')

        return losses

    def run(self) -> None:
        self.initialize_ngf()

        losses = []
        opt = torch.optim.Adam(self.ngf.parameters(), 1e-3)
        for rate in [ 4, 8 ]:
            losses += self.optimize_resolution(opt, rate)

        # Plot results
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme()
        sns.set_palette(sns.color_palette('pastel'))

        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "sans-serif",
        #     "font.sans-serif": "Libertine"
        # })

        plt.plot(losses, label='losses')
        plt.legend()
        plt.yscale('log')
        plt.savefig('losses.pdf')

        logging.info('Finished training neural geometry field')

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
        remap = optext.generate_remapper(self.ngf.complexes.cpu(), cmap, base.shape[0], rate)
        faces = optext.triangulate_shorted(vertices, self.ngf.complexes.shape[0], rate)
        faces = remap.remap_device(faces)

        ps.register_surface_mesh('NGF', vertices.cpu().numpy(), faces.cpu().numpy())
        ps.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str)
    parser.add_argument('--base', type=str)

    args = parser.parse_args()

    trainer = Trainer(args.mesh, args.base)
    trainer.run()
    trainer.display()