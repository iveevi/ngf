import torch
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

        # self.eval = torch.compile(self.eval_kernel, fullgraph=True)

    def eval(self, n):
        normal_array = n.view((-1, 3))
        h_n = torch.nn.functional.pad(normal_array, (0, 1), 'constant', 1.0)
        l = (h_n.t() * (self.M @ h_n.t())).sum(dim=1)
        return l.t().view(n.shape)

def persp_proj(fov_x=45, ar=1, near=0.1, far=100):
    fov_rad = np.deg2rad(fov_x)
    proj_mat = np.array([
        [-1.0 / np.tan(fov_rad / 2.0), 0, 0, 0],
        [0, np.float32(ar) / np.tan(fov_rad / 2.0), 0, 0],
        [0, 0, -(near + far) / (near-far), 2 * far * near / (near-far)],
        [0, 0, 1, 0]
    ])

    proj = torch.tensor(proj_mat, device='cuda', dtype=torch.float32)
    return proj

# NOTE: Adapted from the 'Large Steps in Inverse Rendering' codebase
class Renderer:
    def __init__(self, **config):
        width  = config['width']
        height = config['height']
        near   = config['near']
        far    = config['far']
        aspect = width / height

        self.res  = (height, width)
        self.proj = persp_proj(config['fov'], aspect, near, far)
        self.ctx  = dr.RasterizeCudaContext()
        self.sh   = SphericalHarmonics(config['envmap'])

    def render(self, v, n, f, view_mats):
        mvps = self.proj @ view_mats
        v_hom = torch.nn.functional.pad(v, (0, 1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))
        rast = dr.rasterize(self.ctx, v_ndc, f, self.res)[0]
        light = self.sh.eval(n).contiguous()
        light = dr.interpolate(light, rast, f)[0]
        return dr.antialias(light, rast, v_ndc, f)

    def render_normals(self, v, n, f, view_mats):
        mvps = self.proj @ view_mats
        v_hom = torch.nn.functional.pad(v, (0,1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1,2))
        rast = dr.rasterize(self.ctx, v_ndc, f, self.res)[0]
        col = dr.interpolate(n * 0.5 + 0.5, rast, f)[0]
        bgs = torch.ones_like(col)
        result = dr.antialias(torch.where(rast[..., -1:] != 0, col, bgs), rast, v_ndc, f)
        return result
