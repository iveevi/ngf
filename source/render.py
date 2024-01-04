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

    def render(self, v, n, f, lights, view_mats):
        mvps = self.proj @ view_mats
        v_hom = torch.nn.functional.pad(v, (0, 1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))
        rast = dr.rasterize(self.ctx, v_ndc, f, self.res)[0]
        color = n @ lights.t()
        color = dr.interpolate(color, rast, f)[0]
        return dr.antialias(color, rast, v_ndc, f)

    def render_spherical_harmonics(self, v, n, f, view_mats):
        mvps = self.proj @ view_mats
        v_hom = torch.nn.functional.pad(v, (0, 1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))
        rast = dr.rasterize(self.ctx, v_ndc, f, self.res)[0]
        color = self.sh.eval(n).contiguous()
        color = dr.interpolate(color, rast, f)[0]
        return dr.antialias(color, rast, v_ndc, f)

    def render_normals(self, v, n, f, view_mats):
        mvps = self.proj @ view_mats
        v_hom = torch.nn.functional.pad(v, (0,1), 'constant', 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1,2))
        rast = dr.rasterize(self.ctx, v_ndc, f, self.res)[0]
        col = dr.interpolate(n * 0.5 + 0.5, rast, f)[0]
        bgs = torch.ones_like(col)
        return dr.antialias(torch.where(rast[..., -1:] != 0, col, bgs), rast, v_ndc, f)

def construct_renderer():
    import os
    import imageio

    path = os.path.join(os.path.dirname(__file__), '../images/environment.hdr')
    environment = imageio.imread(path, format='HDR-FI')
    environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
    alpha       = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
    environment = torch.cat((environment, alpha), dim=-1)

    return Renderer(width=1024, height=1024, fov=45.0, near=1e-3, far=1000.0, envmap=environment)

def arrange_camera_views(target):
    import nvdiffrast.torch as dr
    import optext

    context = dr.RasterizeCudaContext()
    proj = persp_proj(45.0, 1.0, 1e-3, 1e3)

    vgraph = optext.vertex_graph(target.faces.cpu())
    vertex_weights = torch.zeros((target.vertices.shape[0], 1), dtype=torch.float32, device='cuda')

    amins = []
    pos = []
    dir = []
    views = []

    zero_count = 0
    while True:
        amin = vertex_weights.argmin()
        amins.append(amin.item())
        amin_vertex = target.vertices[amin]
        amin_normal = target.normals[amin]
        # print('Min coverage vertex', vertex_weights[amin])

        distance = 0.2

        old_coverage = vertex_weights[amin].item()
        if vertex_weights[amin] > 0.99:
            break

        while True:
            eye = amin_vertex + distance * amin_normal
            up = torch.tensor([0, 1, 0], dtype=torch.float32, device='cuda')
            look = -amin_normal
            right = torch.cross(look, up)
            right /= right.norm()
            up = torch.cross(look, right)

            view = torch.tensor([
                [ right[0], up[0], look[0], eye[0] ],
                [ right[1], up[1], look[1], eye[1] ],
                [ right[2], up[2], look[2], eye[2] ],
                [ 0, 0, 0, 1 ]
            ], dtype=torch.float32, device='cuda').inverse()

            mvps = (proj @ view).unsqueeze(0)
            v_hom = torch.nn.functional.pad(target.vertices, (0, 1), 'constant', 1.0)
            v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))

            rasterized = dr.rasterize(context, v_ndc, target.faces, (1024, 1024))[0]
            # image = dr.interpolate(target.normals, rasterized, target.faces)[0]
            # plt.imshow(0.5 + 0.5 * image[0].cpu().numpy())
            # plt.show()

            triangles = rasterized[..., -1].int().flatten() - 1
            nonzero = triangles.nonzero()
            triangles = triangles[nonzero].flatten()
            tris = target.faces[triangles]
            v0s = tris[:, 0].long().unsqueeze(-1)
            v1s = tris[:, 1].long().unsqueeze(-1)
            v2s = tris[:, 2].long().unsqueeze(-1)
            src = torch.ones_like(v0s, dtype=torch.float32)

            copy_weights = vertex_weights.clone()
            copy_weights.scatter_(0, v0s, src)
            copy_weights.scatter_(0, v1s, src)
            copy_weights.scatter_(0, v2s, src)

            current_coverage = copy_weights[amin].item()

            # print('  > coverage delta', current_coverage - old_coverage)
            # if current_coverage - old_coverage < 1e-3:
            #     zero_count += 1

            if current_coverage - old_coverage < 1e-3:
                distance *= 0.7
            else:
                break

        if distance > 0.02:
            views.append(view)

        view_inv = view.inverse()
        view_pos = view_inv @ torch.tensor([0, 0, 0, 1], dtype=torch.float32, device='cuda')
        view_dir = view_inv @ torch.tensor([0, 0, 1, 0], dtype=torch.float32, device='cuda')
        view_pos = view_pos[:3]
        view_dir = view_dir[:3]
        pos.append(view_pos.cpu().numpy())
        dir.append(view_dir.cpu().numpy())

        # TODO: use multiplication with the robust laplacian
        vertex_weights = copy_weights
        vertex_weights = torch.nn.functional.pad(vertex_weights, (0, 2), 'constant', 0.0)
        for i in range(5):
            vertex_weights = vgraph.smooth_device(vertex_weights, 1.0)
        vertex_weights = vertex_weights[:, 0].unsqueeze(-1)

    print('Generated %d views' % len(views), 'Zero count %d' % zero_count)

    views = torch.stack(views)
    print('views', views.shape)

    return views
