import torch
import numpy as np

from .mesh import Mesh


def lerp(X, U, V):
    lp00 = X[:, 0, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * (1.0 - V.unsqueeze(-1))
    lp01 = X[:, 1, :].unsqueeze(1) * U.unsqueeze(-1) * (1.0 - V.unsqueeze(-1))
    lp10 = X[:, 3, :].unsqueeze(1) * (1.0 - U.unsqueeze(-1)) * V.unsqueeze(-1)
    lp11 = X[:, 2, :].unsqueeze(1) * U.unsqueeze(-1) * V.unsqueeze(-1)
    return lp00 + lp01 + lp10 + lp11


def indices(sample_rate):
    triangles = []
    for i in range(sample_rate - 1):
        for j in range(sample_rate - 1):
            a = i * sample_rate + j
            c = (i + 1) * sample_rate + j
            b, d = a + 1, c + 1
            triangles.append([a, b, c])
            triangles.append([b, d, c])

    return np.array(triangles)


def sample_rate_indices(C, sample_rate):
    tri_indices = []
    for i in range(C.shape[0]):
        ind = indices(sample_rate)
        ind += i * sample_rate ** 2
        tri_indices.append(ind)

    tri_indices = np.concatenate(tri_indices, axis=0)
    tri_indices_tensor = torch.from_numpy(tri_indices).int().cuda()
    return tri_indices_tensor


def shorted_indices(V, C, sample_rate=16):
    triangles = []
    for c in range(C.shape[0]):
        offset = c * sample_rate * sample_rate
        for i in range(sample_rate - 1):
            for j in range(sample_rate - 1):
                a = offset + i * sample_rate + j
                c = offset + (i + 1) * sample_rate + j
                b, d = a + 1, c + 1

                vs = V[[a, b, c, d]]
                d0 = np.linalg.norm(vs[0] - vs[3])
                d1 = np.linalg.norm(vs[1] - vs[2])

                if d0 < d1:
                    triangles.append([a, d, b])
                    triangles.append([a, c, d])
                else:
                    triangles.append([a, c, b])
                    triangles.append([b, c, d])

    return np.array(triangles)


def quadify(count, sample_rate=16):
    quads = []
    for c in range(count):
        offset = c * sample_rate * sample_rate
        for i in range(sample_rate - 1):
            for j in range(sample_rate - 1):
                a = offset + i * sample_rate + j
                c = offset + (i + 1) * sample_rate + j
                b, d = a + 1, c + 1
                quads.append([a, b, d, c])

    return np.array(quads)


def make_cmap(complexes, points, LP, sample_rate):
    Cs = complexes.cpu().numpy()
    lp = LP.detach().cpu().numpy()

    cmap = dict()
    for i in range(Cs.shape[0]):
        for j in Cs[i]:
            if cmap.get(j) is None:
                cmap[j] = set()

        corners = np.array([
            0, sample_rate - 1,
            sample_rate * (sample_rate - 1),
            sample_rate ** 2 - 1
        ]) + (i * sample_rate ** 2)

        qvs = points[Cs[i]].cpu().numpy()
        cvs = lp[corners]

        for j in range(4):
            # Find the closest corner
            dists = np.linalg.norm(qvs[j] - cvs, axis=1)
            closest = np.argmin(dists)
            cmap[Cs[i][j]].add(corners[closest])

    return cmap


def average_edge_length(V, T):
    v0 = V[T[:, 0], :]
    v1 = V[T[:, 1], :]
    v2 = V[T[:, 2], :]

    v01 = v1 - v0
    v02 = v2 - v0
    v12 = v2 - v1

    l01 = torch.norm(v01, dim=1)
    l02 = torch.norm(v02, dim=1)
    l12 = torch.norm(v12, dim=1)
    return (l01 + l02 + l12).mean()/3.0


def triangle_areas(v, f):
    v0 = v[f[:, 0], :]
    v1 = v[f[:, 1], :]
    v2 = v[f[:, 2], :]
    v01 = v1 - v0
    v02 = v2 - v0
    return 0.5 * torch.norm(torch.cross(v01, v02, dim=-1), dim=1)


def lookat(eye, center, up):
    normalize = lambda x: x / torch.norm(x)

    f = normalize(eye - center)
    u = normalize(up)
    s = normalize(torch.cross(f, u, dim=-1))
    u = torch.cross(s, f, dim=-1)

    dot_f = torch.dot(f, eye)
    dot_u = torch.dot(u, eye)
    dot_s = torch.dot(s, eye)

    return torch.tensor([
        [s[0], u[0], -f[0], -dot_s],
        [s[1], u[1], -f[1], -dot_u],
        [s[2], u[2], -f[2], dot_f],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device='cuda')


def arrange_views(simplified: Mesh, cameras: int, radius: float = 1.0):
    import ngfutil

    seeds = list(torch.randint(0, simplified.faces.shape[0], (cameras,)).numpy())
    clusters = ngfutil.cluster_geometry(simplified.optg, seeds, 1, 'uniform')

    views = []
    for cluster in clusters:
        faces = simplified.faces[cluster]

        v0 = simplified.vertices[faces[:, 0]]
        v1 = simplified.vertices[faces[:, 1]]
        v2 = simplified.vertices[faces[:, 2]]
        centroid = (v0 + v1 + v2) / 3.0
        centroid = centroid.mean(dim=0)

        normal = torch.cross(v1 - v0, v2 - v0, dim=-1)
        normal = normal.mean(dim=0)
        normal = normal / torch.norm(normal)

        eye = centroid + radius * normal
        up = torch.tensor([0, 1, 0], dtype=torch.float32, device='cuda')
        look = -normal

        if torch.dot(look, up).abs().item() > 1.0 - 1e-6:
            up = torch.tensor([1, 0, 0], dtype=torch.float32, device='cuda')
        if torch.dot(look, up).abs().item() > 1.0 - 1e-6:
            up = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda')

        right = torch.cross(look, up, dim=-1)
        right /= right.norm()

        up = torch.cross(look, right, dim=-1)
        up /= up.norm()

        look /= look.norm()

        view = torch.tensor([
            [ right[0], up[0], look[0], eye[0] ],
            [ right[1], up[1], look[1], eye[1] ],
            [ right[2], up[2], look[2], eye[2] ],
            [ 0, 0, 0, 1 ]
        ], dtype=torch.float32, device='cuda').inverse()

        views.append(view)

    return torch.stack(views)
