import json
import meshio
import os
import sys
import torch
import imageio

from torch.utils.cpp_extension import load
from scripts.geometry import compute_face_normals, compute_vertex_normals

from util import *
from configurations import *

# Generate error figures
# TODO: provide viewing configruations for rendering loss/normal loss...

# Arguments
assert len(sys.argv) == 3, 'evaluate.py <reference> <directory>'
# TODO: directory and /db.sjon

reference = sys.argv[1]
directory = sys.argv[2]

assert reference is not None
assert directory is not None

# Expect specific naming conventions for reference and db
assert os.path.basename(reference) == 'target.obj'
assert os.path.exists(directory)

# Load all necessary extensions
optext = load(name='optext',
        sources=[ 'optext.cu' ],
        extra_include_paths=[ 'glm' ],
        build_directory='build',
        extra_cflags=[ '-O3' ],
        extra_cuda_cflags=[ '-O3' ])

print('Loaded optimization extension')

# Converting meshio to torch
convert = lambda M: (torch.from_numpy(M.points).float().cuda(), torch.from_numpy(M.cells_dict['triangle']).int().cuda())

# Load target reference meshe
print('Loading target mesh:', reference)

target = os.path.basename(reference)
target = meshio.read(reference)

v_ref = torch.from_numpy(target.points).float().cuda()

v_min = torch.min(v_ref, dim=0).values
v_max = torch.max(v_ref, dim=0).values
center = (v_min + v_max) / 2.0
extent = torch.linalg.norm(v_max - v_min) / 2.0
normalize = lambda x: (x - center) / extent
v_ref = normalize(v_ref)

f_ref = torch.from_numpy(target.cells_dict['triangle']).int().cuda()
fn_ref = compute_face_normals(v_ref, f_ref)
vn_ref = compute_vertex_normals(v_ref, f_ref, fn_ref)

# Load an instantiate all the models
models = []

for filename in os.listdir(directory):
    # Only parse .pt
    if not filename.endswith('.pt'):
        continue

    print('Loading model:', filename)

    # Load model
    m = torch.load(os.path.join(directory, filename))
    print(m.keys())
    print('kernel:', m['kernel'])

    m['name'] = os.path.splitext(filename)[0]
    m['kernel'] = sampler(kernel=clerp(lerps[m['kernel']]))
    print('sampler:', m['kernel'])

    model = m['model']
    complexes = m['complexes']
    points = m['points']
    features = m['features']
    smplr    = m['kernel']

    # Sample points
    resolution = 16
    LP, LF = smplr(complexes, points, features, resolution)
    V = model(points=LP, features=LF).detach()

    print('V shape:', V.shape)

    I = shorted_indices(V.cpu().numpy(), complexes, resolution)
    I = torch.from_numpy(I).int()

    cmap = make_cmap(complexes, points, LP, resolution)
    remap = optext.generate_remapper(complexes.cpu(), cmap, LP.shape[0], resolution)
    F = remap.remap(I).cuda()

    # Plot in an image
    import matplotlib.pyplot as plt

    size = np.ceil(np.sqrt(complexes.shape[0])).astype(np.int32)
    print('N:', size)

    # Compute normals
    Fn = compute_face_normals(V, F)
    N = compute_vertex_normals(V, F, Fn)
    print('N shape:', N.shape)

    models.append((V, N, F, m))

# Preferred order of models
ordering = [ 'relu', 'elu', 'siren', 'gauss', 'sinc', 'morlet', 'onion' ]
reorder = lambda x: ordering.index(x[3]['name'].split('-')[0])
models.sort(key=reorder)

# Load scene cameras
from scripts.render import NVDRenderer

scene = {}
# scene['view_mats'] = all_views

# Also load an environment map
environment = imageio.imread('images/environment.hdr', format='HDR-FI')
environment = torch.tensor(environment, dtype=torch.float32, device='cuda')
alpha       = torch.ones((*environment.shape[:2], 1), dtype=torch.float32, device='cuda')
environment = torch.cat((environment, alpha), dim=-1)

scene['res_x'] = 1024
scene['res_y'] = 640
scene['fov'] = 45.0
scene['near_clip'] = 0.1
scene['far_clip'] = 1000.0
# scene['view_mats'] = torch.stack(scene['view_mats'], dim=0)
scene['envmap'] = environment
scene['envmap_scale'] = 1.0

renderer = NVDRenderer(scene)

# Set up a camera in front on the mesh
position = torch.tensor([ -2.0, 0.0, 3.0 ], device='cuda')/2
camera = lookat(position,
    torch.tensor([ 0.0, 0.0, 0.0 ], device='cuda'),
    torch.tensor([ 0.0, 1.0, 0.0 ], device='cuda')
).unsqueeze(0)

# img = renderer.render_normals(v_ref, vn_ref, f_ref, camera)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.rcParams['text.usetex'] = True

def textsc(str):
    return r'\textsc{' + str + '}'

# TODO: TeX formatting...

fig = plt.figure(figsize=(25, 10))

plots = 1 + len(models)

ax = plt.subplot(4, plots, 1)
ref_img = renderer.render(v_ref, vn_ref, f_ref, camera).pow(1.0 / 2.2)[0]
ref_alpha = ref_img[..., 3:]
ref_img = ref_img[..., :3] * ref_alpha + (1.0 - ref_alpha)
# TODO: implement this alpha blending in the rendering...
# ref_img = img.clamp(0.0, 1.0).cpu()
ax.imshow(ref_img.cpu(), origin='lower')
ax.set_title(textsc('Reference'))
ax.axis('off')

# TODO: image grid with comaps...
ax = plt.subplot(4, plots, 1 + 2 * plots)
ref_nrm_img = renderer.render_normals(v_ref, vn_ref, f_ref, camera)[0]
# img = img.clamp(0.0, 1.0).cpu()
ax.imshow(ref_nrm_img.cpu(), origin='lower')
ax.axis('off')

# Metric values in the y axis
# ax = plt.subplot(4, plots, 2 + plots)
# ax.set_ylabel(r'$L_1\;/\;$' + textsc('PSNR'))
#
# ax = plt.subplot(4, plots, 2 + 3 * plots)
# ax.set_ylabel(r'$L_1\;/\;$' + textsc('PSNR'))

# TODO: label with metrics...
from torchmetrics.image import PeakSignalNoiseRatio

psnr = PeakSignalNoiseRatio().cuda()

render_losses = []
normal_losses = []

for i, model in enumerate(models):
    V, N, F, m = model

    name = m['name'].split('-')[0]

    # Render figures
    ax = plt.subplot(4, plots, i + 2)
    img = renderer.render(V, N, F, camera).pow(1.0 / 2.2)[0]
    alpha = img[..., 3:]
    img = img[..., :3] * alpha + (1.0 - alpha)

    ax.imshow(img.cpu(), origin='lower')
    ax.set_title(textsc(name))
    ax.axis('off')

    ax = plt.subplot(4, plots, i + 2 + plots)
    diff = (img - ref_img).abs().mean(dim=-1)
    print('Diff (L2) mean:', diff.mean().item(), diff.shape)

    ax.imshow(diff.cpu(), origin='lower', cmap='coolwarm')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    psnr_value = psnr(img, ref_img)
    print('PSNR:', psnr_value)

    ax.set_xlabel(r'${' + r'{:.3f}\;/\;{:.3f}'.format(diff.mean().item(), psnr_value.item()) + '}$')
    render_losses.append(diff.mean().item())

    # Normal figures
    ax = plt.subplot(4, plots, i + 2 + 2 * plots)
    img = renderer.render_normals(V, N, F, camera)[0]

    ax.imshow(img.cpu(), origin='lower')
    ax.axis('off')

    ax = plt.subplot(4, plots, i + 2 + 3 * plots)
    diff = (img - ref_nrm_img).abs().mean(dim=-1)
    print('Diff (L1) mean:', diff.mean().item())

    ax.imshow(diff.cpu(), origin='lower', cmap='coolwarm')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    psnr_value = psnr(img, ref_nrm_img)
    print('PSNR:', psnr_value)

    ax.set_xlabel(r'${' + r'{:.3f}\;/\;{:.3f}'.format(diff.mean().item(), psnr_value.item()) + '}$')
    normal_losses.append(diff.mean().item())

# Bold the minimum values
render_losses = np.array(render_losses)
normal_losses = np.array(normal_losses)

render_min = np.argmin(render_losses)
normal_min = np.argmin(normal_losses)

ax = plt.subplot(4, plots, render_min + 2 + plots)
clabel = ax.get_xlabel()
ax.set_xlabel(r'$\mathbf{' + clabel[2:-2] + '}$')

ax = plt.subplot(4, plots, normal_min + 2 + 3 * plots)
clabel = ax.get_xlabel()
ax.set_xlabel(r'$\mathbf{' + clabel[2:-2] + '}$')

plt.savefig('figures/activations.pdf', dpi=300, bbox_inches='tight')
plt.show()
