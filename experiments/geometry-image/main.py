import numpy as np
import polyscope as ps

from PIL import Image

gim = Image.open('bunny.p65.gim257.fmp.bmp')
gim.save('gim.png')
gim_np = np.array(gim)
print(gim_np.shape)

print(gim_np[:5, :5, :])

indices = []
size = gim_np.shape[0]
gim_np = gim_np.reshape(-1, 3) / 255.0
for i in range(size - 1):
    for j in range(size - 1):
        ni = i + 1
        nj = j + 1

        vi = gim_np[i * size + j]
        vj = gim_np[i * size + nj]
        vni = gim_np[ni * size + j]
        vnj = gim_np[ni * size + nj]

        # Choose smallest diagonal
        d0 = np.linalg.norm(vi - vnj)
        d1 = np.linalg.norm(vj - vni)

        if d0 < d1:
            indices.append([i * size + j, i * size + nj, ni * size + nj])
            indices.append([i * size + j, ni * size + j, ni * size + nj])
        else:
            indices.append([i * size + j, i * size + nj, ni * size + j])
            indices.append([ni * size + j, i * size + nj, ni * size + nj])

        # indices.append([i * size + j, i * size + nj, ni * size + j])
        # indices.append([ni * size + j, i * size + nj, ni * size + nj])

indices = np.array(indices).reshape(-1, 3)

ps.set_ground_plane_mode('none')

ps.init()
ps_mesh = ps.register_surface_mesh("mesh", gim_np, indices)
# ps_cloud = ps.register_point_cloud("cloud", gim_np)
ps.show()
