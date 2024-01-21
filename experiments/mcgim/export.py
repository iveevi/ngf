import torch
import torchvision
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mcgim', help='path to the target multichart geometry image')

args = parser.parse_args()
mcgim = torch.load(args.mcgim)
print('Shape', mcgim.shape)

def factint(n):
    import numpy as np
    pos_n = abs(n)
    max_candidate = int(np.sqrt(pos_n))
    for candidate in range(max_candidate, 0, -1):
        if pos_n % candidate == 0:
            return candidate, n // candidate
    return n, 1

N, M = factint(mcgim.shape[0])
N, M = M, N
print('Factoring in rectangle of dim (%d, %d)' % (N, M))

sampling = mcgim.shape[1]

flattened = torch.zeros((N * sampling, M * sampling, 3))

for i in range(N):
    for j in range(M):
        subset = mcgim[i * M + j].cpu()
        flattened[i * sampling: (i + 1) * sampling, j * sampling: (j + 1) * sampling] = subset

# print(flattened.shape)
# flattened = 0.5 + 0.5 * flattened / flattened.norm(-1)
# torchvision.utils.save_image(flattened.permute(2, 1, 0), 'mcgim.png')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.figure(figsize=(M/2, N/2))
plt.imshow(flattened[..., 0])
plt.gca().set_position([0, 0, 1, 1])
plt.axis('off')
plt.savefig('mcgim.png')
