import numpy as np
import polyscope as ps

ps.init()

# Original complex
points  = np.array([ [0, 0, 0], [1, 0, 0], [1, 0.5, 1], [0, 0, 1] ], dtype=np.float32)
indices = np.array([ [0, 1, 2, 3] ], dtype=np.uint32)
G = ps.register_surface_mesh('original', points, indices)
G.set_color((0.7, 0.68, 0.9))
G.set_edge_color((0.0, 0.0, 0.0))
G.set_edge_width(2.5)

# Subdivided
rate = 16

subdivided_points = []
for i in range(rate):
    for j in range(rate):
        u, v = i/(rate - 1), j/(rate - 1)

        p0 = u * points[0] + (1 - u) * points[1]
        p1 = u * points[3] + (1 - u) * points[2]
        p = v * p0 + (1 - v) * p1

        # p[2] += 1.5

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
G = ps.register_surface_mesh('subdivided', subdivided_points, subdivided_indices)
G.set_color((0.7, 0.68, 0.9))
G.set_edge_color((0.0, 0.0, 0.0))
G.set_edge_width(1.0)

# And displaced by displacement map
from PIL import Image

img = Image.open('/home/venki/nsc/images/rope.png')
print(img.width, img.height)

for i in range(len(subdivided_points)):
    u, v = subdivided_points[i][0], subdivided_points[i][2]
    x, y = int(u * (img.width - 1)), int(v * (img.height - 1))
    p = subdivided_points[i]
    p[1] += 0.2 * img.getpixel((x, y))[0] / 255.0
    # p[2] += 1.5

G = ps.register_surface_mesh('displaced', subdivided_points, subdivided_indices)
G.set_color((0.7, 0.68, 0.9))
G.set_edge_color((0.0, 0.0, 0.0))
G.set_edge_width(1.0)

ps.set_ground_plane_mode('none')

ps.show()
