import torch
import argparse
import numpy as np

from tqdm import trange

from neural_mcgim import *

parser = argparse.ArgumentParser()
parser.add_argument('--mcgim', help='path to the target multichart geometry image')

args = parser.parse_args()

mcgim = torch.load(args.mcgim)

print('Shape', mcgim.shape)

def factint(n):
    pos_n = abs(n)
    max_candidate = int(np.sqrt(pos_n))
    for candidate in range(max_candidate, 0, -1):
        if pos_n % candidate == 0:
            return candidate, n // candidate
    return n, 1

N, M = factint(mcgim.shape[0])
print('Factoring in rectangle of dim (%d, %d)' % (N, M))

sampling = mcgim.shape[1]

# mcgim = mcgim.reshape((N * sampling, M * sampling, 3))
# print('New shape', mcgim.shape)

textured = torch.zeros((N * sampling, M * sampling, 3)).float().cuda()
for i in range(mcgim.shape[0]):
    x, y = i % N, i // N
    gim = mcgim[i]
    textured[x * sampling : (x + 1) * sampling, y * sampling : (y + 1) * sampling] = gim

# Neural implicit function representing the image
neural_mcgims = ReLU_Feat_Posenc(N, M).cuda()

U = torch.linspace(0, 1, N * sampling)
V = torch.linspace(0, 1, M * sampling)
U, V = torch.meshgrid(U, V, indexing='ij')
UV_whole = torch.stack([U, V], dim=-1).cuda()

opt = torch.optim.Adam(list(neural_mcgims.parameters()), 1e-3)
# sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)

# TODO: CNN architecture as well
losses = []
for i in trange(10_000):
    gims = neural_mcgims(UV_whole)
    loss = (textured - gims).square().mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    # sch.step()

    losses.append(loss.item())

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

plt.rcParams['figure.dpi'] = 600

plt.plot(losses, label='loss')
plt.yscale('log')
plt.savefig('losses.png')

fig, axs = plt.subplots(3, 3, layout='tight')
for i in range(3):
    delta = (textured[..., i] - gims[..., i]).abs()

    axs[0, i].imshow(textured[..., i].cpu().numpy(), aspect='auto')
    axs[1, i].imshow(gims[..., i].detach().cpu().numpy(), aspect='auto')
    axs[2, i].imshow(delta.detach().cpu().numpy(), aspect='auto')

    axs[0, i].axis('off')
    axs[1, i].axis('off')
    axs[2, i].axis('off')

plt.savefig('gims.png')

def display_mcgim(mcgim):
    import polyscope as ps

    ps.init()

    for k in range(N * M):
        x, y = k % N, k // N

        gim = mcgim[x * sampling : (x + 1) * sampling, y * sampling : (y + 1) * sampling]
        gim = gim.reshape(-1, 3).cpu().numpy()

        indices = []
        for i in range(sampling - 1):
            for j in range(sampling - 1):
                a = i * sampling + j
                c = (i + 1) * sampling + j
                b, d = a + 1, c + 1
                indices.append([a, b, c])
                indices.append([b, d, c])

                vs = gim[[a, b, c, d]]
                d0 = np.linalg.norm(vs[0] - vs[3])
                d1 = np.linalg.norm(vs[1] - vs[2])

                if d0 < d1:
                    indices.append([a, b, d])
                    indices.append([a, d, c])
                else:
                    indices.append([a, b, c])
                    indices.append([b, d, c])

        indices = np.array(indices)

        ps.register_surface_mesh('patch-%d' % k, gim, indices)

    ps.show()

import os
filename = 'neural-' + os.path.basename(args.mcgim)
directory = os.path.dirname(args.mcgim)
filename = os.path.join(directory, filename)

print('serializing to', filename)

data = {
    'model': neural_mcgims.eval(),
    'sampling': (N, M, sampling),
}

torch.save(data, filename)

# display_mcgim(textured)
# display_mcgim(gims.detach())
