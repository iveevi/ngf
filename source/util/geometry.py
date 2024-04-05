import torch


def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))


def compute_face_normals(vertices, faces):
    fi = torch.transpose(faces, 0, 1).long()
    vertices = torch.transpose(vertices, 0, 1)

    v = [
        vertices.index_select(1, fi[0]),
        vertices.index_select(1, fi[1]),
        vertices.index_select(1, fi[2])
    ]

    c = torch.cross(v[1] - v[0], v[2] - v[0], dim=0)
    length = torch.linalg.norm(c, dim=0)
    length = torch.where(length == 0, torch.ones_like(length), length)
    return c / length


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

    length = torch.linalg.norm(normals, dim=0)
    length = torch.where(length == 0, torch.ones_like(length), length)
    return (normals / length).transpose(0, 1)


def vertex_normals(vertices, faces):
    face_normals = compute_face_normals(vertices, faces)
    return compute_vertex_normals(vertices, faces, face_normals)
