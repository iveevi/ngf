import torch

def compute_face_normals(verts, faces):
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)

    v = [
        verts.index_select(1, fi[0]),
        verts.index_select(1, fi[1]),
        verts.index_select(1, fi[2])
    ]

    c = torch.cross(v[1] - v[0], v[2] - v[0])
    l = torch.linalg.norm(c, dim=0)
    l = torch.where(l == 0, torch.ones_like(l), l)
    return (c / l)

def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))

def compute_vertex_normals(verts, faces, face_normals):
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)
    normals = torch.zeros_like(verts)

    v = [
        verts.index_select(1, fi[0]),
        verts.index_select(1, fi[1]),
        verts.index_select(1, fi[2])
    ]

    for i in range(3):
        d0 = v[(i + 1) % 3] - v[i]
        d0 = d0 / torch.norm(d0)
        d1 = v[(i + 2) % 3] - v[i]
        d1 = d1 / torch.norm(d1)
        face_angle = safe_acos(torch.sum(d0 * d1, 0))
        nn = face_normals * face_angle
        for j in range(3):
            normals[j].index_add_(0, fi[i], nn[j])

    lengths = torch.linalg.norm(normals, dim=0)
    lengths = torch.where(lengths == 0, torch.ones_like(lengths), lengths)
    return (normals / lengths).transpose(0, 1)

# Vertex density measures
def vertex_density(V, F):
    V0 = V[F[..., 0]]
    V1 = V[F[..., 1]]
    V2 = V[F[..., 2]]

    E0 = V0 - V1
    E1 = V0 - V2
    areas = torch.cross(E0, E1).norm(dim=-1)
    total = areas.sum()/F.shape[0]

    A = torch.zeros(V.shape[0], device=V.device, dtype=V.dtype)

    # TODO: does this scatter add?
    A[F[..., 0]] += areas/3
    A[F[..., 1]] += areas/3
    A[F[..., 2]] += areas/3

    return total/A
