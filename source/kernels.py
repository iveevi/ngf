import torch
import numpy as np

def morlet(x, k):
    return torch.cos(k * np.pi * x) * torch.exp(-k * x ** 2)

import matplotlib.pyplot as plt

u = torch.linspace(0, 1, 1024).cuda()
v = torch.linspace(0, 1, 1024).cuda()
u, v = torch.meshgrid(u, v, indexing='ij')
uv = 1 - 16 * (u * (1 - u) * v * (1 - v))

uvw0 = morlet(uv, 8)
uvw1 = torch.sin(16 * np.pi * uv)

fig, axs = plt.subplots(1, 3)

img0 = axs[0].imshow(uv.cpu().numpy(), cmap='inferno')
plt.colorbar(img0)

img1 = axs[1].imshow(uvw0.cpu().numpy(), cmap='inferno')
plt.colorbar(img1)

img2 = axs[2].imshow(uvw1.cpu().numpy(), cmap='inferno')
plt.colorbar(img2)

axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].set_xticks([])
axs[1].set_yticks([])

plt.show()
