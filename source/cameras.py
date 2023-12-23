import torch
import argparse

from kaolin.ops.conversions import trianglemeshes_to_voxelgrids, voxelgrids_to_cubic_meshes
from kaolin.ops.voxelgrid import fill

from mesh import load_mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str, help='Reference mesh to generate cameras on')

    args = parser.parse_args()

    mesh = args.mesh
    mesh, _ = load_mesh(mesh)

    print('mesh:', mesh.vertices.shape, mesh.faces.shape)

    resolution = 128

    voxels = trianglemeshes_to_voxelgrids(mesh.vertices.unsqueeze(0), mesh.faces, resolution)[0]
    print('voxels', voxels.shape, voxels.sum().item())

    voxels = fill(voxels.unsqueeze(0).cpu())[0].cuda()
    print('voxels', voxels.shape, voxels.sum().item())

    vox_vertices, vox_faces = voxelgrids_to_cubic_meshes(voxels.unsqueeze(0))
    vox_vertices, vox_faces = vox_vertices[0], vox_faces[0]
    vox_vertices /= resolution

    min = torch.min(mesh.vertices, dim=0)[0]
    max = torch.max(mesh.vertices, dim=0)[0]
    scale = torch.max(max - min, dim=0)[0]

    mesh.vertices = (mesh.vertices - min)/scale

    print('origin', min, 'scale', scale)

    print('max vertices', vox_vertices[:, 0].max(), vox_vertices[:, 1].max(), vox_vertices[:, 2].max())

    import polyscope as ps

    ps.init()
    ps.register_surface_mesh('mesh', mesh.vertices.cpu(), mesh.faces.cpu().numpy())
    ps.register_surface_mesh('voxel', vox_vertices.cpu(), vox_faces.cpu().numpy())
    ps.show()
