import argparse
import meshio
import numpy as np
import os
import polyscope as ps
import re
import torch
import optext

from ngf import load_ngf
from util import quadify, shorted_indices
from mesh import load_mesh

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

def preview_single(ngf, refs):
    mode = list(refs.keys())[0] if (len(refs) > 0) else 'ngf'
    def draw(rate, patches):
        ps.remove_all_structures()

        if ngf is not None and mode == 'ngf':
            sample = ngf.sample_uniform(rate)
            V = ngf.eval(*sample).detach()
            print('EVALLED V', V.shape)

            if patches:
                complex_count = ngf.complexes.shape[0]
                V = V.reshape(complex_count, -1, 3).cpu()
                Q = quadify(1, rate)
                for i, patch in enumerate(V):
                    p = ps.register_surface_mesh('patch-%d' % i, patch, Q)
                    p.set_material('wax')
                    # p.set_smooth_shade(True)

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
                F = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)
                m = ps.register_surface_mesh('ngf', V.cpu().numpy(), F.cpu().numpy())
                m.set_color([0.5, 0.5, 1.0])
                m.set_material('wax')

            return

        for r, ref in refs.items():
            if mode == r:
                V, F = ref.vertices, ref.faces
                colors = torch.rand((F.shape[0], 3))
                m = ps.register_surface_mesh(r, V.cpu().numpy(), F.cpu().numpy())
                m.add_color_quantity('false', colors.numpy(), defined_on='faces', enabled=True)
                # m.set_color([0.5, 0.5, 1.0])
                m.set_material('wax')
                # m.set_smooth_shade(True)
                return

    ps.init()

    ps.set_ground_plane_mode('none')

    rate = 4
    patches = True
    def callback():
        import polyscope.imgui as imgui

        nonlocal rate, patches, mode
        imgui.Text('Rate = %d' % rate)
        imgui.SameLine()
        if imgui.Button('Increase'):
            rate = min(32, 2 + rate)
            draw(rate, patches)
        imgui.SameLine()
        if imgui.Button('Decrease'):
            rate = max(2, rate - 2)
            draw(rate, patches)

        imgui.Separator()

        if imgui.Button('Toggle patches'):
            patches = not patches
            draw(rate, patches)

        imgui.Separator()

        for ref in refs:
            if imgui.RadioButton(ref, mode == ref):
                mode = ref
                draw(rate, patches)
                return

        if imgui.RadioButton('NGF Model', mode == 'ngf'):
            mode = 'ngf'
            draw(rate, patches)

    draw(rate, patches)
    ps.set_user_callback(callback)
    ps.show()

def preview_many(many, refs):
    current = list(refs.keys())[0] if (len(refs) > 0) else None
    if current is None:
        list(many.keys())[0]
    def draw(rate):
        ps.remove_all_structures()

        if current in refs:
            V, F = refs[current]
            m = ps.register_surface_mesh('reference', V, F)
            m.set_color([0.5, 0.5, 1.0])
            m.set_material('wax')
            return

        for f, ngf in many.items():
            if current != f:
                continue

            uvs = ngf.sample_uniform(rate)
            V = ngf.eval(*uvs).detach()
            F = optext.triangulate_shorted(V, ngf.complexes.shape[0], rate)

            # Q = quadify(ngf.complexes.shape[0], rate)
            # F = shorted_indices(V, ngf.complexes, rate)

            m = ps.register_surface_mesh(f, V.cpu().numpy(), F.cpu().numpy())
            m.set_color([0.5, 0.5, 1.0])
            m.set_material('wax')
            return

    ps.init()

    ps.set_ground_plane_mode('none')

    rate = 4
    def callback():
        import polyscope.imgui as imgui

        nonlocal rate, current
        imgui.Text('Rate = %d' % rate)
        imgui.SameLine()
        if imgui.Button('Increase'):
            rate = min(32, 2 * rate)
            draw(rate)
        imgui.SameLine()
        if imgui.Button('Decrease'):
            rate = max(2, rate // 2)
            draw(rate)

        imgui.Separator()

        for ref in refs:
            if imgui.RadioButton('Reference', current == ref):
                current = ref
                draw(rate)
                return

        for f in many:
            if imgui.RadioButton(f, current == f):
                current = f
                draw(rate)
                return

    draw(rate)
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

            # Q = quadify(1, rate)
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
    parser.add_argument('--many', type=str, nargs='+', help='path to ngfs')
    parser.add_argument('--references', type=str, nargs='*', help='path to reference mesh')
    parser.add_argument('--lods', type=str, help='directory with lods')
    args = parser.parse_args()

    ngf = None
    if args.ngf is not None:
        assert os.path.exists(args.ngf)

        ngf = torch.load(args.ngf)
        ngf = load_ngf(ngf)

    many = None
    if args.many is not None:
        many = {}
        for file in args.many:
            f = os.path.basename(file).split('.')[0]
            f = os.path.dirname(file) + '-' + f
            n = torch.load(file)
            n = load_ngf(n)
            many[f] = n

    refs = []
    if args.references is not None:
        refs = {}
        for file in args.references:
            mesh, _ = load_mesh(file)
            # refs.append(mesh)
            f = os.path.basename(file)
            f = os.path.dirname(file) + '-' + f
            refs[f] = mesh

    lods = {}
    if args.lods is not None:
        p = re.compile('lod[1-4].pt$')
        for root, _, files in os.walk(args.lods):
            for file in files:
                if p.match(file):
                    base = file.split('.')[0]
                    base = os.path.dirname(file) + '-' + base
                    ngf = torch.load(os.path.join(root, file))
                    ngf = load_ngf(ngf)
                    lods[base] = ngf

    if lods:
        preview_lods(lods)
    if many:
        preview_many(many, refs)
    else:
        assert ngf is not None or refs is not None
        preview_single(ngf, refs)
