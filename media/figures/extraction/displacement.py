import numpy as np
import polyscope as ps

# Original complex
points  = np.array([ [0, 0, 0], [1, 0, 0], [1, 0.5, 1], [0, 0, 1] ], dtype=np.float32)
indices = np.array([ [0, 1, 2, 3] ], dtype=np.uint32)

# Subdivided
rate = 4

subdivided_points = []
for i in range(rate):
    for j in range(rate):
        u, v = i/(rate - 1), j/(rate - 1)

        p0 = u * points[0] + (1 - u) * points[1]
        p1 = u * points[3] + (1 - u) * points[2]
        p = v * p0 + (1 - v) * p1

        subdivided_points.append(p)

subdivided_indices = []
for i in range(rate - 1):
    for j in range(rate - 1):
        i0 = i * rate + j
        i1 = i0 + 1
        i2 = i0 + rate
        i3 = i2 + 1
        subdivided_indices.append([ i0, i1, i3, i2 ])

subdivided_points  = np.array(subdivided_points, dtype=np.float32)
subdivided_indices = np.array(subdivided_indices, dtype=np.uint32)

# And displaced by displacement map
from PIL import Image

img = Image.open('/home/venki/nsc/media/images/rope.png')

def displaced(rate, tri=False):
    displaced_points = []
    for i in range(rate):
        for j in range(rate):
            u, v = i/(rate - 1), j/(rate - 1)

            p0 = u * points[0] + (1 - u) * points[1]
            p1 = u * points[3] + (1 - u) * points[2]
            p = v * p0 + (1 - v) * p1

            x, y = int(u * (img.width - 1)), int(v * (img.height - 1))
            p[1] += 0.1 + 0.2 * img.getpixel((x, y))[0] / 255.0

            displaced_points.append(p)

    displaced_indices = []
    for i in range(rate - 1):
        for j in range(rate - 1):
            i0 = i * rate + j
            i1 = i0 + 1
            i2 = i0 + rate
            i3 = i2 + 1

            if tri:
                displaced_indices.append([ i0, i1, i2 ])
                displaced_indices.append([ i1, i3, i2 ])
            else:
                displaced_indices.append([ i0, i1, i3, i2 ])

    displaced_points  = np.array(displaced_points, dtype=np.float32)
    displaced_indices = np.array(displaced_indices, dtype=np.uint32)

    return displaced_points, displaced_indices

rate = 16
displaced_points, displaced_indices = displaced(rate)

dp_low, di_low = displaced(4, True)

ps.init()

ps.set_ground_plane_mode('none')

og = ps.register_surface_mesh('original', points, indices)
og.set_color([0.5, 0.5, 1.0])
og.set_edge_width(2)

sm = ps.register_surface_mesh('subdivided', subdivided_points, subdivided_indices)
sm.set_color([0.5, 0.5, 1.0])
sm.set_edge_width(2)

dm = ps.register_surface_mesh('displaced low', dp_low, di_low)
dm.set_color([0.5, 0.5, 1.0])
dm.set_edge_width(2)

ogpmesh = ps.register_point_cloud('original', points)
ogpmesh.set_color([0, 0, 0])
ogpmesh.set_radius(0.01)

dpmesh = ps.register_surface_mesh('displaced', displaced_points, displaced_indices)
dpmesh.set_color([0.5, 0.5, 1.0])
dpmesh.set_transparency(0.35)
dpmesh.set_smooth_shade(True)

sp = ps.register_point_cloud('subdivided', subdivided_points)
sp.set_color([0, 0, 0])
sp.set_radius(0.01)

diff = dp_low - subdivided_points
sp.add_vector_quantity('diff', diff, enabled=True, radius=0.005, color=[0.016,0.431,0.149], vectortype='ambient')

dp = ps.register_point_cloud('displaced', dp_low)
dp.set_color([0.129,0.561,0.2])
dp.set_radius(0.01)

ps.set_SSAA_factor(4)

ps.show()
