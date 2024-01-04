# Packages for training
import imageio.v2 as imageio
import meshio
import os
import sys
import robust_laplacian
import torch
import json
import optext
import tqdm

from render import Renderer, arrange_camera_views
from geometry import compute_vertex_normals, compute_face_normals

from util import *
from configurations import *

from ngf import NGF
from mesh import Mesh, load_mesh, simplify_mesh

def arrange_views(simplified: Mesh, cameras: int):
    seeds = list(torch.randint(0, simplified.faces.shape[0], (cameras,)).numpy())
    clusters = optext.cluster_geometry(simplified.optg, seeds, 3, 'uniform')

    cluster_centroids = []
    cluster_normals = []

    for cluster in clusters:
        faces = simplified.faces[cluster]

        v0 = simplified.vertices[faces[:, 0]]
        v1 = simplified.vertices[faces[:, 1]]
        v2 = simplified.vertices[faces[:, 2]]
        centroids = (v0 + v1 + v2) / 3.0
        centroids = centroids.mean(dim=0)


        normals = torch.cross(v1 - v0, v2 - v0)
        normals = normals.mean(dim=0)
        normals = normals / torch.norm(normals)

        cluster_centroids.append(centroids)
        cluster_normals.append(normals)

    cluster_centroids = torch.stack(cluster_centroids, dim=0)
    cluster_normals = torch.stack(cluster_normals, dim=0)

    # Generate camera views
    canonical_up = torch.tensor([0.0, 1.0, 0.0], device='cuda')
    cluster_eyes = cluster_centroids + cluster_normals * 1.0
    cluster_ups = torch.stack(len(clusters) * [ canonical_up ], dim=0)
    # cluster_rights = torch.cross(cluster_normals, cluster_ups)
    # cluster_ups = torch.cross(cluster_rights, cluster_normals)

    all_views = [ lookat(eye, view_point, up) for eye, view_point, up in zip(cluster_eyes, cluster_centroids, cluster_ups) ]
    all_views = torch.stack(all_views, dim=0)

    return all_views

def construct_renderer():
    path = os.path.join(os.path.dirname(__file__), '../images/environment.hdr')
    environment = imageio.imread(path, format='HDR-FI')
    environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
    alpha       = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
    environment = torch.cat((environment, alpha), dim=-1)
    return Renderer(width=1024, height=1024, fov=45.0, near=1e-3, far=1000.0, envmap=environment)

if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Usage: python train.py <config>'

    config = sys.argv[1]
    print('Using configuration: {}'.format(config))

    # Load configuration
    with open(config, 'r') as f:
        config = json.load(f)

    # Load mesh
    target_path    = config['target']
    result_path     = config['directory']
    clusters        = config['clusters']
    batch_size      = config['batch_size']
    resolution      = config['resolution']
    experiments     = config['experiments']

    global_batch_size = batch_size

    print('Loading target mesh from {}'.format(target_path))
    target, normalizer = load_mesh(target_path)
    print('target: {} vertices, {} faces'.format(target.vertices.shape[0], target.faces.shape[0]))

    # simplified = simplify_mesh(target, 10_000, normalizer)
    # print('simplified: {} vertices, {} faces'.format(simplified.vertices.shape[0], simplified.faces.shape[0]))

    # Generate cameras
    # print('geometry of simplified: {}'.format(simplified.optg))
    # views = arrange_views(simplified, clusters)
    # print('views: {} cameras'.format(views.shape[0]))

    arrange_camera_views(target)

    # renderer = construct_renderer()
    #
    # import nvdiffrast.torch as dr
    # import matplotlib.pyplot as plt
    # import polyscope as ps
    #
    # # Compute the mean curvature on the surface
    # # L, M = robust_laplacian.mesh_laplacian(
    # #     target.vertices.cpu().numpy(),
    # #     target.faces.cpu().numpy()
    # # )
    #
    # # np.reciprocal(M.data, out=M.data)
    # # print('mass', M)
    #
    # vertices = target.vertices.cpu().numpy()
    # normals = target.normals.cpu().numpy()
    #
    # vgraph = optext.vertex_graph(target.faces.cpu())
    # snormals = vgraph.smooth_device(target.normals, 1.0)
    #
    # # snormals = L @ normals
    #
    # # import scipy.sparse.linalg
    # # snormals = scipy.sparse.linalg.spsolve(L, normals)
    #
    # # L = M @ L
    #
    # # print('laplacian', L.shape)
    # # mean_curvature = L @ vertices
    # # mean_curvature = np.linalg.norm(mean_curvature, axis=-1) / 2.0
    # # print('mean C', mean_curvature.shape)
    #
    # vertex_weights = torch.zeros((target.vertices.shape[0], 1), dtype=torch.float32, device='cuda')
    #
    # amins = []
    # pos = []
    # dir = []
    # views = []
    #
    # # for i in range(20):
    #
    # # TODO: if failure, then decrease distances
    # zero_count = 0
    # while True:
    #     # TODO: get smoothed normals
    #     # after doing taubin smoothing etc
    #     amin = vertex_weights.argmin()
    #     amins.append(amin.item())
    #     amin_vertex = target.vertices[amin]
    #     amin_normal = target.normals[amin]
    #     print('Min coverage vertex', vertex_weights[amin])
    #
    #     distance = 0.25
    #
    #     old_coverage = vertex_weights[amin].item()
    #     if vertex_weights[amin] > 0.99:
    #         break
    #
    #     while True:
    #         eye = amin_vertex + distance * amin_normal
    #
    #         up = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda')
    #         look = -amin_normal
    #         right = torch.cross(look, up)
    #         right /= right.norm()
    #         up = torch.cross(look, right)
    #
    #         view = torch.tensor([
    #             [ right[0], up[0], look[0], eye[0] ],
    #             [ right[1], up[1], look[1], eye[1] ],
    #             [ right[2], up[2], look[2], eye[2] ],
    #             [ 0, 0, 0, 1 ]
    #         ], dtype=torch.float32, device='cuda').inverse()
    #
    #         mvps = (renderer.proj @ view).unsqueeze(0)
    #         v_hom = torch.nn.functional.pad(target.vertices, (0, 1), 'constant', 1.0)
    #         v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))
    #
    #         rasterized = dr.rasterize(renderer.ctx, v_ndc, target.faces, (1024, 1024))[0]
    #         image = dr.interpolate(target.normals, rasterized, target.faces)[0]
    #         # plt.imshow(0.5 + 0.5 * image[0].cpu().numpy())
    #         # plt.show()
    #
    #         triangles = rasterized[..., -1].int().flatten() - 1
    #         nonzero = triangles.nonzero()
    #         triangles = triangles[nonzero].flatten()
    #         tris = target.faces[triangles]
    #         v0s = tris[:, 0].long().unsqueeze(-1)
    #         v1s = tris[:, 1].long().unsqueeze(-1)
    #         v2s = tris[:, 2].long().unsqueeze(-1)
    #         src = torch.ones_like(v0s, dtype=torch.float32)
    #
    #         copy_weights = vertex_weights.clone()
    #         copy_weights.scatter_(0, v0s, src)
    #         copy_weights.scatter_(0, v1s, src)
    #         copy_weights.scatter_(0, v2s, src)
    #
    #         current_coverage = copy_weights[amin].item()
    #
    #         print('  > coverage delta', current_coverage - old_coverage)
    #
    #         # if current_coverage - old_coverage < 1e-3:
    #         #     zero_count += 1
    #
    #         if current_coverage - old_coverage < 1e-3:
    #             distance *= 0.7
    #         else:
    #             break
    #
    #     if distance > 0.05:
    #         views.append(view)
    #
    #     view_inv = view.inverse()
    #     view_pos = view_inv @ torch.tensor([0, 0, 0, 1], dtype=torch.float32, device='cuda')
    #     view_dir = view_inv @ torch.tensor([0, 0, 1, 0], dtype=torch.float32, device='cuda')
    #     view_pos = view_pos[:3]
    #     view_dir = view_dir[:3]
    #     pos.append(view_pos.cpu().numpy())
    #     dir.append(view_dir.cpu().numpy())
    #
    #     # TODO: use multiplication with the robust laplacian
    #     vertex_weights = copy_weights
    #     vertex_weights = torch.nn.functional.pad(vertex_weights, (0, 2), 'constant', 0.0)
    #     for i in range(5):
    #         vertex_weights = vgraph.smooth_device(vertex_weights, 1.0)
    #     vertex_weights = vertex_weights[:, 0].unsqueeze(-1)
    #
    # print('Generated %d views' % len(views), 'Zero count %d' % zero_count)
    #
    # # ps.init()
    # # m = ps.register_surface_mesh('target', vertices, target.faces.cpu().numpy())
    # # m.add_vector_quantity('normals', target.normals.cpu().numpy())
    # # m.add_vector_quantity('smoothed normals', snormals.cpu().numpy())
    # # m.add_scalar_quantity('wsum', vertex_weights.flatten().cpu().numpy(), enabled=True)
    # # ps.register_point_cloud('pc', vertices[amins])
    # # ps.register_point_cloud('pos', np.array(pos)).add_vector_quantity('dir', np.array(dir), enabled=True)
    # # ps.show()
    #
    # views = torch.stack(views)
    # print('views', views.shape)
    #
    # N = int(np.sqrt(len(views)))
    # M = (len(views) + N - 1) // N
    #
    # fig, axs = plt.subplots(N, M)
    # axs = axs.flatten()
    #
    # for ax, view in zip(axs, views):
    #     image = renderer.render_normals(target.vertices, target.normals, target.faces, view.unsqueeze(0))[0]
    #     ax.imshow(image.cpu().numpy())
    #     ax.axis('off')
    #
    # plt.show()
