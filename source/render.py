import os
import torch
import numpy as np
import imageio
import nvdiffrast.torch as dr


def persp_proj(fov_x, ar, near, far):
    fov_rad = np.deg2rad(fov_x)
    proj_mat = np.array([
        [-1.0 / np.tan(fov_rad / 2.0), 0, 0, 0],
        [0, np.float32(ar) / np.tan(fov_rad / 2.0), 0, 0],
        [0, 0, -(near + far) / (near-far), 2 * far * near / (near-far)],
        [0, 0, 1, 0]
    ])

    return torch.tensor(proj_mat, device='cuda', dtype=torch.float32)


# NOTE: Adapted from the 'Large Steps in Inverse Rendering' codebase
class Renderer:
    ENVIRONMENT = os.path.join(os.path.dirname(__file__), os.path.pardir, 'media', 'environment.hdr')

    def __init__(self,
                 width: int = 256,
                 height: int = 256,
                 fov: float = 45.0,
                 near:float = 0.1,
                 far: float = 1000.0,
                 environment: str = ENVIRONMENT) -> None:
        self.res = (height, width)
        self.proj = persp_proj(fov, width/height, near, far)
        self.ctx = dr.RasterizeCudaContext()

    def render(self, v, n, f, views):
        mvps = self.proj @ views
        v_hom = torch.nn.functional.pad(v, (0, 1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))

        layers = []
        with dr.DepthPeeler(self.ctx, v_ndc, f, self.res) as peeler:
            for i in range(2):
                rast, rast_db = peeler.rasterize_next_layer()

                vertices = dr.interpolate(v, rast, f)[0]
                vertices = dr.antialias(vertices, rast, v_ndc, f)

                normals = dr.interpolate(n, rast, f)[0]
                normals = dr.antialias(normals, rast, v_ndc, f)

                layers += [vertices, normals]

        return torch.concat(layers, dim=-1)
