import os
import torch
import logging
import numpy as np
import nvdiffrast.torch as dr


class SphericalHarmonics:
    def __init__(self, envmap):
        h, w = envmap.shape[:2]

        # Compute the grid of theta, phi values
        theta = (torch.linspace(0, np.pi, h, device='cuda')).repeat(w, 1).t()
        phi = (torch.linspace(3*np.pi, np.pi, w, device='cuda')).repeat(h,1)

        # Compute the value of sin(theta) once
        sin_theta = torch.sin(theta)
        # Compute x,y,z
        # This differs from the original formulation as here the up axis is Y
        x = sin_theta * torch.cos(phi)
        z = -sin_theta * torch.sin(phi)
        y = torch.cos(theta)

        # Compute the polynomials
        Y_0 = 0.282095

        # The following are indexed so that using Y_n[-p]...Y_n[p] gives the proper polynomials
        Y_1 = [
            0.488603 * z,
            0.488603 * x,
            0.488603 * y
            ]

        Y_2 = [
            0.315392 * (3*z.square() - 1),
            1.092548 * x*z,
            0.546274 * (x.square() - y.square()),
            1.092548 * x*y,
            1.092548 * y*z
        ]

        area = w * h
        radiance = envmap[..., :3]
        dt_dp = 2.0 * np.pi ** 2 / area

        # Compute the L coefficients
        L = [
            [ (radiance * Y_0 * (sin_theta)[..., None] * dt_dp).sum(dim=(0,1)) ],
            [ (radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0,1)) for y in Y_1 ],
            [ (radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0,1)) for y in Y_2 ],
        ]

        # Compute the R,G and B matrices
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743125
        c4 = 0.886227
        c5 = 0.247708

        self.M = torch.stack([
            torch.stack([ c1 * L[2][2] , c1 * L[2][-2], c1 * L[2][1] , c2 * L[1][1]           ]),
            torch.stack([ c1 * L[2][-2], -c1 * L[2][2], c1 * L[2][-1], c2 * L[1][-1]          ]),
            torch.stack([ c1 * L[2][1] , c1 * L[2][-1], c3 * L[2][0] , c2 * L[1][0]           ]),
            torch.stack([ c2 * L[1][1] , c2 * L[1][-1], c2 * L[1][0] , c4 * L[0][0] - c5 * L[2][0]])
        ]).movedim(2,0)

    def eval(self, n):
        normal_array = n.view((-1, 3))
        h_n = torch.nn.functional.pad(normal_array, (0, 1), 'constant', 1.0)
        l = (h_n.t() * (self.M @ h_n.t())).sum(dim=1)
        return l.t().view(n.shape)


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
                 # width: int = 512,
                 # height: int = 512,
                 fov: float = 45.0,
                 near:float = 0.1,
                 far: float = 1000.0) -> None:
        self.res = (height, width)
        self.proj = Renderer.projection(fov, width/height, near, far)
        self.ctx = dr.RasterizeCudaContext()

        import imageio
        environment = imageio.v2.imread(Renderer.ENVIRONMENT, format='HDR')
        environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
        alpha = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
        environment = torch.cat((environment, alpha), dim=-1)
        self.sh = SphericalHarmonics(environment)

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

        # rast = dr.rasterize(self.ctx, v_ndc, f, self.res)[0]
        # normals = dr.interpolate(n, rast, f)[0]
        # return dr.antialias(normals, rast, v_ndc, f)

    def shaded(self, v, n, f, view_mats):
        mvps = self.proj @ view_mats
        v_hom = torch.nn.functional.pad(v, (0, 1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))
        rast = dr.rasterize(self.ctx, v_ndc, f, self.res)[0]
        color = self.sh.eval(n).contiguous()
        color = dr.interpolate(color, rast, f)[0]
        return dr.antialias(color, rast, v_ndc, f)

    @torch.no_grad()
    def interpolate(self, v, a, f, view_mats):
        mvps = self.proj @ view_mats
        v_hom = torch.nn.functional.pad(v, (0, 1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))
        rast = dr.rasterize(self.ctx, v_ndc, f, self.res)[0]
        return dr.interpolate(a, rast, f)[0]
