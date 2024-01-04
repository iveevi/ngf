import torch

def compute_face_normals(verts, faces):
    """
    Compute per-face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)

    v = [verts.index_select(1, fi[0]),
                 verts.index_select(1, fi[1]),
                 verts.index_select(1, fi[2])]

    # assert not torch.isnan(v[0]).any()
    # assert not torch.isnan(v[1]).any()
    # assert not torch.isnan(v[2]).any()

    c = torch.cross(v[1] - v[0], v[2] - v[0])
    # assert not torch.isnan(c).any()
    l = torch.linalg.norm(c, dim=0)
    l = torch.where(l == 0, torch.ones_like(l), l)
    # assert not (l == 0).any()

    return (c / l)

def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))

def compute_vertex_normals(verts, faces, face_normals):
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)
    normals = torch.zeros_like(verts)

    v = [verts.index_select(1, fi[0]),
             verts.index_select(1, fi[1]),
             verts.index_select(1, fi[2])]

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
