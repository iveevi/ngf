import argparse
import meshio
import numpy as np
import os
import polyscope as ps
import re
import torch

from ngf import load_ngf
from util import quadify


COLOR_WHEEL = [
        np.array([0.880, 0.320, 0.320]),
        np.array([0.880, 0.530, 0.320]),
        np.array([0.880, 0.740, 0.320]),
        np.array([0.810, 0.880, 0.320]),
        np.array([0.600, 0.880, 0.320]),
        np.array([0.390, 0.880, 0.320]),
        np.array([0.320, 0.880, 0.460]),
        np.array([0.320, 0.880, 0.670]),
        np.array([0.320, 0.880, 0.880]),
        np.array([0.320, 0.670, 0.880]),
        np.array([0.320, 0.460, 0.880]),
        np.array([0.390, 0.320, 0.880]),
        np.array([0.600, 0.320, 0.880]),
        np.array([0.810, 0.320, 0.880]),
        np.array([0.880, 0.320, 0.740]),
        np.array([0.880, 0.320, 0.530])
]
    
def preview_single(ngf, ref):
    mode = 'ref'
    def draw(rate, patches):
        ps.remove_all_structures()

        if ngf is not None and mode == 'ngf':
            V = ngf.eval(rate).detach().cpu()

            complex_count = ngf.complexes.shape[0]
            V = V.reshape(complex_count, -1, 3)

            Q = quadify(1, rate)
            for i, patch in enumerate(V):
                p = ps.register_surface_mesh('patch-%d' % i, patch, Q)
                p.set_material('wax')
                p.set_smooth_shade(True)

                if patches:
                    color = COLOR_WHEEL[i % len(COLOR_WHEEL)]
                    p.set_color(color)

                    # Get boundary vertices in order
                    boundary = []

                    for j in range(rate):
                        boundary.append(patch[j])

                    for j in range(rate):
                        boundary.append(patch[j * rate + rate - 1])

                    for j in range(rate - 1, -1, -1):
                        boundary.append(patch[(rate - 1) * rate + j])

                    for j in range(rate - 1, -1, -1):
                        boundary.append(patch[j * rate])

                    indices = []
                    for j in range(len(boundary)):
                        nj = (j + 1) % len(boundary)
                        indices.append((j, nj))

                    boundary = torch.stack(boundary, dim=0).numpy()
                    indices = np.array(indices)

                    c = ps.register_curve_network('boundary-%d' % i, boundary, indices)
                    c.set_color([0, 0, 0])
                    c.set_radius(0.0025)
                else:
                    p.set_color([0.6, 0.5, 0.9])

        if ref is not None and mode == 'ref':
            V, F = ref
            m = ps.register_surface_mesh('reference', V, F)
            m.set_color([0.6, 0.5, 0.9])
            m.set_material('wax')
            m.set_smooth_shade(True)

    ps.init()

    ps.set_ground_plane_mode('none')

    rate = 4
    patches = True
    def callback():
        import polyscope.imgui as imgui

        nonlocal rate, patches
        if imgui.Button('Increase'):
            rate = min(16, 2 * rate)
            draw(rate, patches)
        imgui.SameLine()
        if imgui.Button('Decrease'):
            rate = max(2, rate // 2)
            draw(rate, patches)

        imgui.Separator()

        if imgui.Button('Toggle patches'):
            patches = not patches
            draw(rate, patches)

        imgui.Separator()

        global mode
        if imgui.RadioButton('Reference', mode == 'ref'):
            mode = 'ref'
            draw(rate, patches)
        elif imgui.RadioButton('NGF Model', mode == 'ngf'):
            mode = 'ngf'
            draw(rate, patches)

    draw(rate, patches)
    ps.set_user_callback(callback)
    ps.show()

def preview_lods(lods):
    current = list(lods.keys())[0]
    def draw(rate, patches):
        ps.remove_all_structures()

        for name, ngf in lods.items():
            if current != name:
                continue

            V = ngf.eval(rate).detach().cpu()

            complex_count = ngf.complexes.shape[0]
            V = V.reshape(complex_count, -1, 3)

            Q = quadify(1, rate)
            for i, patch in enumerate(V):
                p = ps.register_surface_mesh('patch-%d' % i, patch, Q)
                p.set_material('wax')
                # p.set_smooth_shade(True)

                if patches:
                    color = COLOR_WHEEL[i % len(COLOR_WHEEL)]
                    p.set_color(color)

                    # Get boundary vertices in order
                    boundary = []

                    for j in range(rate):
                        boundary.append(patch[j])

                    for j in range(rate):
                        boundary.append(patch[j * rate + rate - 1])

                    for j in range(rate - 1, -1, -1):
                        boundary.append(patch[(rate - 1) * rate + j])

                    for j in range(rate - 1, -1, -1):
                        boundary.append(patch[j * rate])

                    indices = []
                    for j in range(len(boundary)):
                        nj = (j + 1) % len(boundary)
                        indices.append((j, nj))

                    boundary = torch.stack(boundary, dim=0).numpy()
                    indices = np.array(indices)

                    c = ps.register_curve_network('boundary-%d' % i, boundary, indices)
                    c.set_color([0, 0, 0])
                    c.set_radius(0.0025)
                else:
                    p.set_color([0.6, 0.5, 0.9])
    
    ps.init()

    ps.set_ground_plane_mode('none')

    rate = 4
    patches = True
    def callback():
        import polyscope.imgui as imgui

        nonlocal rate, patches, current
        if imgui.Button('Increase'):
            rate = min(16, 2 * rate)
            draw(rate, patches)
        imgui.SameLine()
        if imgui.Button('Decrease'):
            rate = max(2, rate // 2)
            draw(rate, patches)

        imgui.Separator()

        if imgui.Button('Toggle patches'):
            patches = not patches
            draw(rate, patches)

        imgui.Separator()
        for name in lods:
            if imgui.RadioButton(name, name == current):
                current = name
                draw(rate, patches)
                break

    draw(rate, patches)
    ps.set_user_callback(callback)
    ps.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngf', type=str, help='path to ngf')
    parser.add_argument('--reference', type=str, help='path to reference mesh')
    parser.add_argument('--lods', type=str, help='directory with lods')
    args = parser.parse_args()

    ngf = None
    if args.ngf is not None:
        assert os.path.exists(args.ngf)

        ngf = torch.load(args.ngf)
        ngf = load_ngf(ngf)

    ref = None
    if args.reference is not None:
        assert os.path.exists(args.reference)
        mesh = meshio.read(args.reference)

        V = mesh.points
        F = mesh.cells_dict['triangle']

        max_v = np.max(V, axis=0)
        min_v = np.min(V, axis=0)
        
        center = (min_v + max_v) / 2
        extent = np.sqrt(np.sum((max_v - min_v) ** 2)) / 2.0
        
        V = (V - center) / extent
        ref = (V, F)

    lods = {}
    if args.lods is not None:
        p = re.compile('lod[1-4].pt$')
        for root, _, files in os.walk(args.lods):
            for file in files:
                if p.match(file):
                    base = file.split('.')[0]
                    ngf = torch.load(os.path.join(root, file))
                    ngf = load_ngf(ngf)
                    lods[base] = ngf

    if lods:
        preview_lods(lods)
    else:
        assert ngf is not None or ref is not None
        preview_single(ngf, ref)
