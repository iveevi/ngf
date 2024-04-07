import os
import torch
import numpy as np
import nvdiffrast.torch as dr


class Renderer:
    ENVIRONMENT = os.path.join(os.path.dirname(__file__), os.path.pardir, 'media', 'environment.hdr')

    @staticmethod
    def projection(fov: float, ar: float, near: float, far: float) -> torch.Tensor:
        fov_rad = np.deg2rad(fov)
        proj_mat = np.array([
            [-1.0 / np.tan(fov_rad / 2.0), 0, 0, 0],
            [0, np.float32(ar) / np.tan(fov_rad / 2.0), 0, 0],
            [0, 0, -(near + far) / (near-far), 2 * far * near / (near-far)],
            [0, 0, 1, 0]
        ])

        return torch.tensor(proj_mat, device='cuda', dtype=torch.float32)

    def __init__(self,
                 width: int = 256,
                 height: int = 256,
                 fov: float = 45.0,
                 near:float = 0.1,
                 far: float = 1000.0) -> None:
        self.res = (height, width)
        self.proj = Renderer.projection(fov, width/height, near, far)
        self.ctx = dr.RasterizeCudaContext()

    def render(self, v: torch.Tensor, n: torch.Tensor, f: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        mvps = self.proj @ views
        v_hom = torch.nn.functional.pad(v, (0, 1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))

        layers = []
        with dr.DepthPeeler(self.ctx, v_ndc, f, self.res) as peeler:
            for i in range(3):
                rast, rast_db = peeler.rasterize_next_layer()
                normals = dr.interpolate(n, rast, f)[0]
                normals = dr.antialias(normals, rast, v_ndc, f)
                layers += [normals]

        return torch.concat(layers, dim=-1)
