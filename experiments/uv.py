import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

res = 1024
U = np.linspace(0, 1, res)
V = np.linspace(0, 1, res)

U, V = np.meshgrid(U, V)

U_plus, U_minus = U, 1 - U
V_plus, V_minus = V, 1 - V
UV = 16 * U_plus * U_minus * V_plus * V_minus

ftns = [
        lambda u, v, uv: uv,
        lambda u, v, uv: 1 - uv,
        # lambda u, v, uv: np.sin(8 * uv),
        # lambda u, v, uv: np.sin(8 * (1 - uv)),
        lambda u, v, uv: np.sin(2 ** 6 * np.sqrt(np.abs(u - 0.5) ** 2 + np.abs(v - 0.5) ** 2)),
        lambda u, v, uv: np.cos(2 ** 6 * np.sqrt(np.abs(u - 0.5) ** 2 + np.abs(v - 0.5) ** 2)),
        lambda u, v, uv: np.sin(2 ** 6 * np.fmod(np.abs(u - 0.5) * np.abs(v - 0.5), 1/2 ** 3)),
        lambda u, v, uv: np.cos(2 ** 3 * np.exp(np.cos(2 ** 3 * np.abs(u - 0.5)) + np.cos(2 ** 3 * np.abs(v - 0.5))))
]

fig, axs = plt.subplots(1, len(ftns))

for f, ax in zip(ftns, axs):
    im = ax.imshow(f(U, V, UV), vmin=0, vmax=1, cmap='hot')
    ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([])

plt.show()
