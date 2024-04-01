import os
import torch

from render import Renderer
from ngf import load_ngf

def new_renderer():
    import imageio.v2 as imageio
    path = os.path.join(os.path.dirname(__file__), '../images/environment.hdr')
    path = os.path.abspath(path)
    environment = imageio.imread(path, format='HDR')
    environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
    alpha       = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
    environment = torch.cat((environment, alpha), dim=-1)
    return Renderer(width=1280, height=720, fov=45.0, near=0.1, far=1000.0, envmap=environment)
    # return Renderer(width=512, height=512, fov=30.0, near=0.1, far=1000.0, envmap=environment)

def view_for(tag):
    # Predefined locations
    predefined = {
            'xyz'        : torch.tensor([ 2, 0, 1 ], device='cuda').float(),
            'einstein'   : torch.tensor([ 0, 0, 3.5 ], device='cuda').float(),
            'skull'      : torch.tensor([ -0.5, 0, 2.5 ], device='cuda').float(),
            'armadillo'  : torch.tensor([ -1.8, 0, -2.8 ], device='cuda').float(),
            'nefertiti'  : torch.tensor([ 0, 0, -3.5 ], device='cuda').float(),
            'lucy'       : torch.tensor([ 0, 0, -4.0 ], device='cuda').float(),
            'dragon'     : torch.tensor([ 0, 0, 3.5 ], device='cuda').float(),
            'indonesian' : torch.tensor([ 3 * 0.2, 0, -4 * 0.2 ], device='cuda').float(),
    }

    # Defaults
    eye    = torch.tensor([ 0, 0, 3 ], device='cuda').float()
    up     = torch.tensor([ 0, 1, 0 ], device='cuda').float()
    center = torch.tensor([ 0, 0, 0 ], device='cuda').float()

    if tag in predefined:
        eye = predefined[tag]
    else:
        print('No config specified for', tag)

    look = center - eye
    if torch.dot(look, up).abs().item() > 1.0 - 1e-6:
        up = torch.tensor([1, 0, 0], device='cuda').float()
    if torch.dot(look, up).abs().item() > 1.0 - 1e-6:
        up = torch.tensor([0, 0, 1], device='cuda').float()

    right = torch.cross(look, up)
    right /= right.norm()

    up = torch.cross(look, right)
    up /= up.norm()

    look /= look.norm()

    return torch.tensor([
        [ right[0], up[0], look[0], eye[0] ],
        [ right[1], up[1], look[1], eye[1] ],
        [ right[2], up[2], look[2], eye[2] ],
        [ 0, 0, 0, 1 ]
    ], dtype=torch.float32, device='cuda').inverse()

def ngf_to_mesh(ngf, rate=16):
    import optext
    from util import make_cmap
    from mesh import mesh_from
    with torch.no_grad():
        uvs = ngf.sample_uniform(rate)
        V = ngf.eval(*uvs)
        base = ngf.base(rate)
    cmap = make_cmap(ngf.complexes, ngf.points.detach(), base, rate)
    remap = optext.generate_remapper(ngf.complexes.cpu(), cmap, base.shape[0], rate)
    indices = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
    F = remap.remap_device(indices)
    return mesh_from(V, F)

# TODO: update benchmarks after mlp.eval
def cropbox(images):
    import numpy as np
    boxes = []
    for image in images.values():
        rgb = image[:,:,:3].cpu().numpy()
        sums = np.sum(rgb, axis=2)
        mask = np.where(sums > 0, 1, 0)

        indices = np.where(mask == 1)
        rgb_box = (np.min(indices[1]), np.min(indices[0]), np.max(indices[1]), np.max(indices[0]))

        boxes.append(rgb_box)

    left = min([ box[0] for box in boxes ])
    top = min([ box[1] for box in boxes ])
    right = max([ box[2] for box in boxes ])
    bottom = max([ box[3] for box in boxes ])

    return (left, right, top, bottom)

if __name__ == '__main__':
    import argparse
    import torchvision

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str)
    parser.add_argument('--ngf', type=str, nargs='+', help='path to ngf for thumbnail')

    args = parser.parse_args()

    renderer = new_renderer()
    view = view_for(args.tag).unsqueeze(0)
    # print('tag', args.tag, args.ngf)

    colors = {}

    images = {}
    for ngf in args.ngf:
        print('ngf', ngf)
        prefix = os.path.basename(ngf).split('.')[0]

        ngf = load_ngf(torch.load(ngf))
        ngf_mesh = ngf_to_mesh(ngf)

        F = ngf_mesh.faces.shape[0]
        C = None
        if F in colors:
            C = colors[F]
        else:
            C = 0.5 + 0.5 * torch.randn((F, 3), device='cuda')
            colors[F] = C

        img = renderer.render_spherical_harmonics(ngf_mesh.vertices, ngf_mesh.normals, ngf_mesh.faces, view)[0]
        # img = renderer.render_false_coloring(ngf_mesh.vertices, ngf_mesh.normals, ngf_mesh.faces, C, view)[0]
        images[prefix] = img

        import matplotlib.pyplot as plt
        plt.imshow(img.cpu().numpy())
        plt.show()

    directory = os.path.join('media', 'thumbnails')
    abs_directory = os.path.abspath(directory)
    os.makedirs(abs_directory, exist_ok=True)

    cbox = cropbox(images)

    for k, img in images.items():
        img = img.permute(2, 0, 1)

        # img = (img/800).pow(1/2.2)
        img = img[:, cbox[2] : cbox[3] + 1, cbox[0] : cbox[1] + 1]
        alpha = (img.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ img, alpha ], dim=0)
        print('image', img.shape)

        pimg = os.path.join(directory, k + '.png')
        torchvision.utils.save_image(img, pimg)
