import argparse
import figuregen
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import simpleimageio as sio
import torch

from figuregen.util.image import Cropbox, relative_mse
from figuregen.util.templates import CropComparison

# TeX presents
preamble = r'''
\documentclass[varwidth=100cm, border=0pt]{standalone}

\renewcommand{\familydefault}{\sfdefault}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{caption}
\usepackage{graphicx}
\usepackage{libertine}
\usepackage{multirow}
\usepackage{pgfplots}
\usepackage{subcaption}
\usepackage{tikz}
\usepackage{array}
\usepackage{tabularray}
\usepackage{multirow}

\usetikzlibrary{calc}

\definecolor{color0}{HTML}{619cce}
\definecolor{color1}{HTML}{bb903e}
\definecolor{color2}{HTML}{8767c9}
\definecolor{color3}{HTML}{69a657}
\definecolor{color4}{HTML}{c75d9e}
\definecolor{color5}{HTML}{cb584d}

\pgfplotsset{compat=1.18}
\pgfplotsset{
    yticklabel style={
        /pgf/number format/fixed,
        /pgf/number format/precision=8
    },
    scaled y ticks=false
}
'''

# legend style={draw=none, fill=none, at={(0.5, -0.4)}, anchor=north},
# legend style={fill=none},
# title style={font=\large},
lineplot_template = r'''\begin{tikzpicture}[baseline=(current bounding box.north)]
\begin{axis}[
    ymode=log,
    title=%s,
    name=%s, %s
    xshift=2cm,
    width=%.2f cm,
    height=%.2f cm,
    ylabel=%s,
    xlabel=%s,
    legend columns=%d,
    legend style={draw=none, fill=none, at={(0.5, -0.4)}, anchor=north},
    xmin=%.3f,
    xmax=%.3f,
    ymin=%.3f,
    ymax=%.3f,
    xtick distance=%d,
    grid=major,
    grid style={dashed, gray!30},
]

%s

\end{axis}
\end{tikzpicture}'''

def fill_lineplot(**kwargs):
    return lineplot_template % (kwargs['title'], kwargs['codename'],
                                kwargs['loc'], kwargs['width'],
                                kwargs['height'], kwargs['ylabel'],
                                kwargs['xlabel'], 3,
                                kwargs['x_min'], kwargs['x_max'],
                                kwargs['y_min'], kwargs['y_max'],
                                kwargs['tick_step'], kwargs['plots'])

def plot_marked(key, d, color, legend):
    line = r'\addplot [mark=*, line width=2pt, color=%s] coordinates {' % color

    for v in d:
        line += '(%f, %f) ' % v
    line += '};\n'
    if legend:
        line += r'\label{plot:' + key + '}'
        line += r'\addlegendentry{\textsc{' + key + '}}'

    return line

def plot_transparent_end(key, d, color, legend):
    line = r'\addplot [line width=1pt, color=%s, opacity=0.65] coordinates {' % color

    for v in d:
        line += '(%f, %f) ' % v
    line += '};\n'
    if legend:
        line += r'\label{plot:' + key + '}'
        line += r'\addlegendentry{\textsc{' + key + '}}'

    line += '\\addplot [mark=*, color=%s, forget plot] coordinates { (%f, %f) };' % (color, d[-1][0], d[-1][1])

    return line

def lineplot(data, name, xlabel='X', ylabel='', width=8, height=6, mode='marked', at=None, legend=False):
    codename = name.lower().replace(' ', '-')
    title = '\\textsc{%s}' % name

    # Generate LaTeX code
    colors = [ 'color0', 'color1', 'color2', 'color3', 'color4', 'color5' ]

    lines = []
    for color, (key, d) in zip(colors, data.items()):
        if mode == 'marked':
            lines.append(plot_marked(key, d, color, legend))
        elif mode == 'transparent':
            lines.append(plot_transparent_end(key, d, color, legend))
        else:
            raise NotImplementedError

    x_min = min([ min([ v[0] for v in d ]) for d in data.values() ])
    x_max = max([ max([ v[0] for v in d ]) for d in data.values() ])
    stride = (x_max - x_min) / 25
    x_min = max(0, x_min - stride)
    x_max += stride

    y_min = min([ min([ v[1] for v in d ]) for d in data.values() ])
    y_max = max([ max([ v[1] for v in d ]) for d in data.values() ])
    print('yrange:', y_min, y_max)

    stride = (y_max - y_min) / 25
    y_min -= stride
    y_max += stride

    print('--> yrange:', y_min, y_max)

    loc = '' if at is None else 'at={(%s)},' % at
    addplot = '\n'.join(lines)

    tex = fill_lineplot(title=title, codename=codename, loc=loc, width=width,
                        height=height, xlabel=xlabel, ylabel=ylabel,
                        num_cols=len(data), x_min=x_min, x_max=x_max,
                        y_min=y_min, y_max=y_max, tick_step=args.tick,
                        plots=addplot)

    return codename, tex

def synthesize_tex(code, filename, log=True):
    import tempfile
    import subprocess

    # Write to temporary file
    with tempfile.NamedTemporaryFile('w+') as fp:
        print('Writing to', fp.name)
        fp.write(code)
        fp.seek(0)

        os.makedirs(os.path.dirname('resources/figures/generated/'), exist_ok=True)

        if log:
            subprocess.check_call(['xelatex', '-interaction=nonstopmode', fp.name],
                cwd=os.path.dirname('resources/figures/generated/'))
        else:
            subprocess.check_call(['xelatex', '-interaction=nonstopmode', fp.name],
                cwd=os.path.dirname('resources/figures/generated/'), stdout=subprocess.DEVNULL)

        # Copy resulting PDF to destination
        result = os.path.join('resources/figures/generated/', os.path.basename(fp.name) + '.pdf')
        subprocess.check_call(['cp', result, filename])
        print('Retrieving', result, 'into', filename)

# Batch crop
def cropbox(images):
    boxes = []
    for image in images.values():
        rgb = image[:,:,:3].cpu().numpy()
        sums = np.sum(rgb, axis=2)
        mask = np.where(sums > 0, 1, 0)

        indices = np.where(mask == 1)
        rgb_box = (np.min(indices[1]), np.min(indices[0]), np.max(indices[1]), np.max(indices[0]))

        boxes.append(rgb_box)

    left = min([ box[0] for box in boxes ])
    top = min([ box[1] for box in boxes ])
    right = max([ box[2] for box in boxes ])
    bottom = max([ box[3] for box in boxes ])

    return (left, right, top, bottom)

# Results plots (gather from directory)
def results_plot(name, db):
    import torchvision

    print('scene', name, 'database', db.keys())

    # Get closest to 1000 patches
    # counts = np.array(list(db['Ours'].keys()))
    # index = np.argmin(np.abs(1000 - counts))
    # primary = counts[index]
    primary, mind = 0, 1e10
    for i, entry in enumerate(db['Ours']):
        if mind > abs(entry['count'] - 1000):
            mind = abs(entry['count'] - 1000)
            primary = i

    directory = os.path.join('resources', 'figures', 'generated')
    abs_directory = os.path.abspath(directory)
    os.makedirs(abs_directory, exist_ok=True)

    primary_ngf = db['Ours'][primary]

    primary_size = primary_ngf['size']

    primary_qslim_index, mind = 0, 1e10
    for i, entry in enumerate(db['QSlim']):
        if mind > abs(entry['size'] - primary_size):
            mind = abs(entry['size'] - primary_size)
            primary_qslim_index = i

    primary_qslim = db['QSlim'][primary_qslim_index]
    primary_nvdiff = db['nvdiffmodeling'][primary_qslim_index]

    print('Ours count', primary_ngf['count'])

    images = {
            'render:ref' : primary_ngf['images']['render:ref'],
            'normal:ref' : primary_ngf['images']['normal:ref'],

            'render:ngf' : primary_ngf['images']['render:mesh'],
            'normal:ngf' : primary_ngf['images']['normal:mesh'],

            'render:qslim' : primary_qslim['images']['render:mesh'],
            'normal:qslim' : primary_qslim['images']['normal:mesh'],

            'render:nvdiff' : primary_nvdiff['images']['render:mesh'],
            'normal:nvdiff' : primary_nvdiff['images']['normal:mesh'],
    }

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axs = plt.subplots(2, 4)

    for ax, img in zip(axs[0], [ images['render:ref'], images['render:ngf'], images['render:qslim'], images['render:nvdiff'] ]):
        ax.imshow(img.cpu().numpy())

    for ax, img in zip(axs[1], [ images['normal:ref'], images['normal:ngf'], images['normal:qslim'], images['normal:nvdiff'] ]):
        ax.imshow(img.cpu().numpy())

    for ax in axs.flatten():
        ax.axis('off')

    plt.show()
    plt.savefig('images.png')

    # Whitespace removal cropbox
    cbox = cropbox(images)

    # Inset cropboxes for all scenes
    #   ( left, right, top, bottom )
    inset_cropboxes = {
        'armadillo' : (240, 340, 120, 220),
        'dragon'    : (0, 150, 500, 650),
        'metatron'  : (300, 500, 50, 250),
        'nefertiti' : (150, 250, 300, 500),
        'skull'     : (520, 720, 180, 330),
        'venus'     : (100, 250, 300, 600),
        'xyz'       : (300, 500, 250, 400),
        'lucy'      : (100, 200, 300, 480),
        'einstein'  : (300, 450, 250, 400),
    }

    incbox = inset_cropboxes[name]

    for k, img in images.items():
        pimg = os.path.join(directory, k.replace(':', '-') + '.png')

        img = img.permute(2, 0, 1)
        img = img[ :, cbox[2] : cbox[3] + 1, cbox[0] : cbox[1] + 1]

        if 'render' in k:
            img = (img/5).pow(1/2.2)

        alpha = (img.sum(dim=0) > 0).unsqueeze(0)
        print('alpha shape', alpha.shape, alpha.sum())

        img = torch.concat([ img, alpha ], dim=0)
        print('new image shape', img.shape)

        print('saving image', k, img.shape)

        # And the inset as well
        pinset = os.path.join(directory, k.replace(':', '-') + '-inset.png')
        inset = img[ :, incbox[2] : incbox[3] + 1, incbox[0] : incbox[1] + 1]

        torchvision.utils.save_image(img, pimg)
        torchvision.utils.save_image(inset, pinset)

        box = torch.tensor([ incbox[0], incbox[2], incbox[1], incbox[3] ]).unsqueeze(0)
        img = torchvision.io.read_image(pimg)
        rgb = img[:3]
        rgb = torchvision.utils.draw_bounding_boxes(rgb, boxes=box, colors="red", width=8)
        rgb = rgb / 255

        alpha = (rgb.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ rgb, alpha ], dim=0)
        torchvision.utils.save_image(img, pimg)

    # Collect lineplot data
    render_data = {}
    normal_data = {}
    chamfer_data = {}

    for entry in db['Ours']:
        render_data.setdefault('ours', []).append((entry['cratio'], entry['render']))
        normal_data.setdefault('ours', []).append((entry['cratio'], entry['normal']))
        chamfer_data.setdefault('ours', []).append((entry['cratio'], entry['chamfer']))

    for entry in db['QSlim']:
        render_data.setdefault('qslim', []).append((entry['cratio'], entry['render']))
        normal_data.setdefault('qslim', []).append((entry['cratio'], entry['normal']))
        chamfer_data.setdefault('qslim', []).append((entry['cratio'], entry['chamfer']))

    for entry in db['nvdiffmodeling']:
        render_data.setdefault('nvdiffmodeling', []).append((entry['cratio'], entry['render']))
        normal_data.setdefault('nvdiffmodeling', []).append((entry['cratio'], entry['normal']))
        chamfer_data.setdefault('nvdiffmodeling', []).append((entry['cratio'], entry['chamfer']))

    for key in [ 'ours', 'qslim', 'nvdiffmodeling' ]:
        render_data[key] = sorted(render_data[key], key=lambda x: x[0])
        normal_data[key] = sorted(normal_data[key], key=lambda x: x[0])
        chamfer_data[key] = sorted(chamfer_data[key], key=lambda x: x[0])

    print('render data', render_data)
    print('render data', chamfer_data)

    # # Generate lineplots
    _, code0 = lineplot(render_data, '', ylabel=r'\large \textsc{Render}', xlabel='{}', width=6, height=4)
    _, code2 = lineplot(chamfer_data, '', ylabel=r'\large \textsc{Chamfer}', xlabel='Compression ratio', width=6, height=4, legend=True)

    combined = code0 + '\n' + code2

    # Create the code
    round_ten = lambda x: 5 * ((x + 4) // 5)

    code = preamble + r'''
    \begin{document}
        \setlength{\fboxsep}{0pt}
        \setlength{\fboxrule}{1pt}

        %%\begin{tabular}[H]{ccccccc}
        \begin{tblr}{
                stretch=0,
                cells={valign=m, halign=c},
                row{1}={bg=blue!25, font=\Large, rowsep=5pt},
                column{7}={bg=white, font=\normalsize},
                colspec={ccccccc},
        }
            { \textsc{Reference} } & { \textsc{NGF (Ours)} } & { \textsc{Reference} } & { \textsc{NGF (Ours)} } & { \textsc{QSlim} } & { \textsc{nvdiffmodeling} }
            & \SetCell[r=6]{c}
            \begin{minipage}{7cm}
                %s
            \end{minipage} \\
            & & & { \large $%.0f\times$ Compression } & { \large $%.0f\times$ Compression } & { \large $%.0f\times$ Compression } & \\
            \SetCell[r=2]{c} \includegraphics[width=3cm]{%s} &
            \SetCell[r=2]{c} \includegraphics[width=3cm]{%s} &
            \fbox{\includegraphics[width=3cm]{%s}} &
            \fbox{\includegraphics[width=3cm]{%s}} &
            \fbox{\includegraphics[width=3cm]{%s}} &
            \fbox{\includegraphics[width=3cm]{%s}} & \\
            & & \large $\mathcal{L}_1$ & \large %.4f & \large %.4f & \large %.4f & \\
            \SetCell[r=2]{c} \includegraphics[width=3cm]{%s} &
            \SetCell[r=2]{c} \includegraphics[width=3cm]{%s} &
            \fbox{\includegraphics[width=3cm]{%s}} &
            \fbox{\includegraphics[width=3cm]{%s}} &
            \fbox{\includegraphics[width=3cm]{%s}} &
            \fbox{\includegraphics[width=3cm]{%s}} & \\
            & & \large $\mathcal{L}_1$ & \large %.4f & \large %.4f & \large %.4f & \\
        \end{tblr}
        %%\end{tabular}
    \end{document}''' % (
            combined,

            round_ten(primary_ngf['cratio']),
            round_ten(primary_qslim['cratio']),
            round_ten(primary_nvdiff['cratio']),

            abs_directory + '/render-ref.png',
            abs_directory + '/render-ngf.png',
            abs_directory + '/render-ref-inset.png',
            abs_directory + '/render-ngf-inset.png',
            abs_directory + '/render-qslim-inset.png',
            abs_directory + '/render-nvdiff-inset.png',

            primary_ngf['render'],
            primary_qslim['render'],
            primary_nvdiff['render'],

            abs_directory + '/normal-ref.png',
            abs_directory + '/normal-ngf.png',
            abs_directory + '/normal-ref-inset.png',
            abs_directory + '/normal-ngf-inset.png',
            abs_directory + '/normal-qslim-inset.png',
            abs_directory + '/normal-nvdiff-inset.png',

            primary_ngf['normal'],
            primary_qslim['normal'],
            primary_nvdiff['normal'],
    )

    print('CODE', code)

    synthesize_tex(code, os.path.join('resources', 'figures', name + '.pdf'))

# Loss plots (gather from directory)
def loss_plot(dir):
    # Global parameters
    plt.rcParams['text.usetex'] = True

    # Gather all .losses files
    files = [ f for f in os.listdir(dir) if f.endswith('.csv') ]

    strip = lambda s: ''.join(s.split('.')[0].split('-')[:-1])
    lines = lambda f: open(os.path.join(dir, f), 'r').readlines()
    combine = lambda lines: ''.join(lines)
    csv = lambda f: combine(lines(f)).split(',')
    pointed = lambda f: [ (i, np.log10(float(l))) for i, l in enumerate(csv(f)) ]

    graphs = { strip(f) : pointed(f) for f in files }

    _, tex = lineplot(graphs, 'Loss', ylabel='Loss', xlabel='Iterations',
                      width=12, height=7, mode='transparent', legend=True)

    tex = document_template % tex
    synthesize_tex(tex, 'losses.pdf')

# Tabl generation
def table(dbs):
    from prettytable import PrettyTable

    exclude = [ 'Nefertiti' ]
    # exclude = [ ]

    def round_nines(x, k):
        if k == 'Ours':
            a = np.array([ 100, 250, 1000, 2500 ])
            i = np.argmin(np.abs(a - x))
            return a[i]
            # return int(np.ceil(x/10) * 10)
        else:
            return int(np.ceil(x/1000) * 1000)

    converted = {}
    for scene, db in dbs.items():
        if scene in exclude:
            continue

        print('Scene', scene)
        for k, ddb in db.items():
            print('  >', k)
            if k == 'reference':
                converted.setdefault('size', {})
                converted['size'][scene] = ddb
            else:
                # converted.setdefault(k, {})
                for kk in ddb:
                    count = round_nines(kk['count'], k)
                    # print('     >', kk, count)
                    t = (k, count)
                    # converted[k].setdefault(count, {})
                    converted.setdefault(t, {})
                    converted[t][scene] = kk

    # print('converted', converted)
    # converted_list = [ (k, v) for k, v in converted.items() ]
    # print(converted_list)

    # print(converted)
    ours_converted = []
    qslim_converted = []
    nvdiffmodeling_converted = []

    for k, db in converted.items():
        # print(k, '->', db)
        if type(k) is tuple:
            if k[0] == 'Ours':
                ours_converted.append((k, db))
            if k[0] == 'QSlim':
                qslim_converted.append((k, db))
            if k[0] == 'nvdiffmodeling':
                nvdiffmodeling_converted.append((k, db))

    ours_converted.sort(key=lambda t: t[0][1])
    ours_converted = { t[0]: t[1] for t in ours_converted }

    qslim_converted.sort(key=lambda t: t[0][1])
    qslim_converted = { t[0]: t[1] for t in qslim_converted }

    nvdiffmodeling_converted.sort(key=lambda t: t[0][1])
    nvdiffmodeling_converted = { t[0]: t[1] for t in nvdiffmodeling_converted }

    tbl = PrettyTable()

    # tbl.field_names = [] # [ 'Primitives', 'Size' ]

    field_names = [ '', 'Primitives', 'Size' ]
    for scene in dbs:
        if scene in exclude:
            continue

        field_names.append(scene)

    tbl.field_names = field_names

    # Raw sizes
    sizes = []
    for f in field_names:
        if f == '':
            sizes.append('Raw mesh size')
            continue

        if f in [ 'Primitives', 'Size' ]:
            sizes.append('')
            continue

        sizem = converted['size'][f]
        sizemb = sizem['size'] / 1024 ** 2
        # sizetris = sizem['count']

        # sizes.append('%.2f MB / %dK' % (sizemb, sizetris / 1000))
        sizes.append('%.2f MB' % sizemb)

    tbl.add_row(sizes, divider=True)

    # Number of triangles
    sizes = []
    for f in field_names:
        if f == '':
            sizes.append('Triangle count')
            continue

        if f in [ 'Primitives', 'Size' ]:
            sizes.append('')
            continue

        sizem = converted['size'][f]
        sizetris = sizem['count']

        if sizetris > 1_000_000:
            sizes.append('%.1f M' % (sizetris / 1_000_000))
        else:
            sizes.append('%.1f K' % (sizetris / 1_000))

    tbl.add_row(sizes, divider=True)

    # Each method
    to_add = []
    for t in ours_converted:
        # if not t[0] == 'Ours':
        #     continue

        print('t', t)

        data = converted[t]

        sizem = [ d['size'] for d in list(data.values()) ]
        sizem = int(sum(sizem)/len(sizem))

        row = []
        for f in field_names:
            # ref_sizem = converted['size'][f]
            # cratio = ref_sizem/sizem
            if f == '':
                row.append('Ours')
                continue
            if f == 'Primitives':
                row.append(t[1])
                continue
            if f == 'Size':
                row.append('%d KB' % (sizem // 1024))
                continue

            if not f in data:
                row.append('')
                continue

            d = data[f]
            # row.append('%.2f / %.2f / %.2f' % (1e2 * d['render'], 1e2 * d['normal'], 1e5 * d['chamfer']))
            row.append('%.2f / %.2f' % (1e2 * d['render'], 1e5 * d['chamfer']))

        to_add.append(row)

    for i in range(len(to_add)):
        if i == len(to_add) - 1:
            tbl.add_row(to_add[i], divider=True)
        else:
            tbl.add_row(to_add[i])

    to_add = []
    for t in qslim_converted:
        data = converted[t]

        sizem = [ d['size'] for d in list(data.values()) ]
        sizem = max(sizem)

        row = []
        for f in field_names:
            if f == '':
                row.append('QSlim')
                continue
            if f == 'Primitives':
                row.append(t[1])
                continue
            if f == 'Size':
                row.append('%d KB' % (sizem // 1024))
                continue

            if not f in data:
                row.append('')
                continue

            d = data[f]
            # row.append('%.2f / %.2f / %.2f' % (1e2 * d['render'], 1e2 * d['normal'], 1e5 * d['chamfer']))
            row.append('%.2f / %.2f' % (1e2 * d['render'], 1e5 * d['chamfer']))

        to_add.append(row)

    for i in range(len(to_add)):
        if i == len(to_add) - 1:
            tbl.add_row(to_add[i], divider=True)
        else:
            tbl.add_row(to_add[i])

    for t in nvdiffmodeling_converted:
        data = converted[t]

        sizem = [ d['size'] for d in list(data.values()) ]
        sizem = max(sizem)

        row = []
        for f in field_names:
            if f == '':
                row.append('nvdiffmodeling')
                continue
            if f == 'Primitives':
                row.append(t[1])
                continue
            if f == 'Size':
                row.append('%d KB' % (sizem // 1024))
                continue

            if not f in data:
                row.append('')
                continue

            d = data[f]
            # row.append('%.2f / %.2f / %.2f' % (1e2 * d['render'], 1e2 * d['normal'], 1e5 * d['chamfer']))
            row.append('%.2f / %.2f' % (1e2 * d['render'], 1e5 * d['chamfer']))

        # print(row)
        tbl.add_row(row)

    print(tbl.get_latex_string())

def tessellation(db):
    render = {}
    normal = {}
    chamfer = {}

    for scene in db:
        print('scene', scene)
        render_line = []
        normal_line = []
        chamfer_line = []

        for entry in db[scene]:
            render_error = db[scene][entry]['render']
            normal_error = db[scene][entry]['normal']
            chamfer_error = db[scene][entry]['chamfer']

            entry = int(entry)

            render_line.append((entry, render_error))
            normal_line.append((entry, normal_error))
            chamfer_line.append((entry, chamfer_error))

        render[scene] = render_line
        normal[scene] = normal_line
        chamfer[scene] = chamfer_line

    primary = db[args.primary]

    images = {
            'ref'    : primary[2]['ref'],
            'tess2'  : primary[2]['mesh'],
            'tess4'  : primary[4]['mesh'],
            'tess8'  : primary[8]['mesh'],
            'tess12' : primary[12]['mesh'],
            'tess16' : primary[16]['mesh'],
    }

    # Whitespace removal cropbox
    cbox = cropbox(images)

    import torchvision

    directory     = os.path.join('resources', 'figures', 'generated')
    abs_directory = os.path.abspath(directory)

    for k, img in images.items():
        pimg = os.path.join(directory, k + '.png')

        img = img.permute(2, 0, 1)
        img = img[ :, cbox[2] : cbox[3] + 1, cbox[0] : cbox[1] + 1]
        img = (img/5).pow(1/2.2)

        alpha = (img.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ img, alpha ], dim=0)
        torchvision.utils.save_image(img, pimg)

    print(args.primary, primary.keys())

    _, render_code = lineplot(render, 'Render', xlabel='Tessellation', ylabel='{}', width=6, height=5)
    _, normal_code = lineplot(normal, 'Normal', xlabel='Tessellation', ylabel='{}', width=6, height=5)
    _, chamfer_code = lineplot(chamfer, '', xlabel='Tessellation', ylabel=r'\textsc{Chamfer}', width=11, height=5, legend=True)

    combined = render_code + '\n' + normal_code + '\n\n' + chamfer_code

    code = preamble + r'''
    \captionsetup[subfigure]{justification=centering}

    \begin{document}
        \setlength{\fboxsep}{0pt}
        \setlength{\fboxrule}{1pt}

        \begin{tblr}{ colspec={cccc}, cells={valign=t, halign=c} }
            \begin{minipage}{3cm}
                \centering
                \textsc{Resolution %d}\vspace{1mm}
                \includegraphics[width=\textwidth]{%s}
            \end{minipage}
            & \begin{minipage}{3cm}
                \centering
                \textsc{Resolution %d}\vspace{1mm}
                \includegraphics[width=\textwidth]{%s}
            \end{minipage}
            & \begin{minipage}{3cm}
                \centering
                \textsc{Resolution %d}\vspace{1mm}
                \includegraphics[width=\textwidth]{%s}
            \end{minipage}
            & \SetCell[r=2]{c} \begin{minipage}{14cm} \centering %s \end{minipage} \\
            \begin{minipage}{3cm}
                \centering
                \textsc{Resolution %d}\vspace{1mm}
                \includegraphics[width=\textwidth]{%s}
            \end{minipage}
            & \begin{minipage}{3cm}
                \centering
                \textsc{Resolution %d}\vspace{1mm}
                \includegraphics[width=\textwidth]{%s}
            \end{minipage}
            & \begin{minipage}{3cm}
                \centering
                \textsc{Reference}\vspace{1mm}
                \includegraphics[width=\textwidth]{%s}
            \end{minipage}
            & \\
        \end{tblr}
    \end{document}''' % (
        2, abs_directory + '/tess2.png',
        4, abs_directory + '/tess4.png',
        8, abs_directory + '/tess8.png',
        combined,
        12, abs_directory + '/tess12.png',
        16, abs_directory + '/tess16.png',
        abs_directory + '/ref.png'
    )

    synthesize_tex(code, 'tessellation.pdf')

def features(db):
    render = {}
    normal = {}
    chamfer = {}

    exclude = { 'skull' }
    for scene in db:
        if scene in exclude:
            continue

        print('scene', scene)
        render_line = []
        normal_line = []
        chamfer_line = []

        entries = db[scene]
        entries = list(entries.items())
        entries = [ (int(f), d) for f, d in entries ]
        entries = sorted(entries, key=lambda v: v[0])
        if len(entries) < 4:
            continue

        for f, data in entries:
            render_error = data['render']
            normal_error = data['normal']
            chamfer_error = data['chamfer']
            size = data['size'] // 1024

            render_line.append((size, render_error))
            normal_line.append((size, normal_error))
            chamfer_line.append((size, chamfer_error))

        render[scene] = render_line
        normal[scene] = normal_line
        chamfer[scene] = chamfer_line

    # Plots
    n0, render_code = lineplot(render, 'Render', xlabel='Size (KB)', ylabel='Error', width=5, height=4)
    n1, normal_code = lineplot(normal, 'Normal', xlabel='Size (KB)', ylabel='', width=5, height=4)
    _,  chamfer_code = lineplot(chamfer, '', xlabel='Size (KB)', ylabel='Chamfer', height=4, legend=True)

    combined = render_code + '\n' + normal_code + '\n\n' + chamfer_code

    # Load the images
    primary = db[args.primary]

    images = {
            'ref'    : primary[5]['ref'],
            'feat5'  : primary[5]['mesh'],
            'feat10'  : primary[10]['mesh'],
            'feat20'  : primary[20]['mesh'],
            'feat50'  : primary[50]['mesh'],
    }

    # Whitespace removal cropbox
    cbox = cropbox(images)

    cbox1 = 100, 200, 250, 350
    cbox2 = 100, 200, 50, 150

    import torchvision

    directory     = os.path.join('resources', 'figures', 'generated')
    abs_directory = os.path.abspath(directory)

    for k, img in images.items():
        img = img.permute(2, 0, 1)
        img = img[ :, cbox[2] : cbox[3] + 1, cbox[0] : cbox[1] + 1]
        # img = (img/5).pow(1/2.2)

        alpha = (img.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ img, alpha ], dim=0)

        # And the insets as well
        inset1 = img[ :, cbox1[2] : cbox1[3] + 1, cbox1[0] : cbox1[1] + 1]
        inset2 = img[ :, cbox2[2] : cbox2[3] + 1, cbox2[0] : cbox2[1] + 1]

        pimg = os.path.join(directory, k + '.png')
        pinset1 = os.path.join(directory, k.replace(':', '-') + '-inset1.png')
        pinset2 = os.path.join(directory, k.replace(':', '-') + '-inset2.png')

        torchvision.utils.save_image(inset1, pinset1)
        torchvision.utils.save_image(inset2, pinset2)
        torchvision.utils.save_image(img, pimg)

        box1 = torch.tensor([ cbox1[0], cbox1[2], cbox1[1], cbox1[3] ]).unsqueeze(0)
        box2 = torch.tensor([ cbox2[0], cbox2[2], cbox2[1], cbox2[3] ]).unsqueeze(0)

        img = torchvision.io.read_image(pimg)
        rgb = img[:3]
        rgb = torchvision.utils.draw_bounding_boxes(rgb, boxes=box1, colors="red", width=8)
        rgb = torchvision.utils.draw_bounding_boxes(rgb, boxes=box2, colors="blue", width=8)
        rgb = rgb / 255

        alpha = (rgb.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ rgb, alpha ], dim=0)
        torchvision.utils.save_image(img, pimg)

    print(args.primary, primary.keys())

    code = preamble + r'''
    \captionsetup[subfigure]{justification=centering}

    \begin{document}
        \setlength{\fboxsep}{0pt}
        \setlength{\fboxrule}{1pt}

        \begin{figure}
            \centering
            \begin{tblr}{ colspec={cccccc}, cells={valign=t, halign=c} }
                \SetCell[r=3]{c}
                \begin{minipage}{5cm}
                    \centering
                    \textsc{Reference}
                    \vspace{1mm}
                    \includegraphics[width=5cm]{%s}
                \end{minipage}
                & \textsc{5 Features} & \textsc{10 Features} & \textsc{20 Features} & \textsc{50 Features}
                & \SetCell[r=3]{c} \begin{minipage}{10cm} \centering %s \end{minipage} \\
                & \fbox{\includegraphics[width=3cm]{%s}}
                & \fbox{\includegraphics[width=3cm]{%s}}
                & \fbox{\includegraphics[width=3cm]{%s}}
                & \fbox{\includegraphics[width=3cm]{%s}} & \\
                & \fbox{\includegraphics[width=3cm]{%s}}
                & \fbox{\includegraphics[width=3cm]{%s}}
                & \fbox{\includegraphics[width=3cm]{%s}}
                & \fbox{\includegraphics[width=3cm]{%s}} & \\
            \end{tblr}
        \end{figure}
    \end{document}''' % (
        abs_directory + '/ref.png',
        combined,
        abs_directory + '/feat5-inset1.png',
        abs_directory + '/feat10-inset1.png',
        abs_directory + '/feat20-inset1.png',
        abs_directory + '/feat50-inset1.png',
        abs_directory + '/feat5-inset2.png',
        abs_directory + '/feat10-inset2.png',
        abs_directory + '/feat20-inset2.png',
        abs_directory + '/feat50-inset2.png'
    )

    synthesize_tex(code, 'features.pdf')

def mutlichart(db):
    data = {}
    for k, d in db.items():
        for x in d:
            data.setdefault(k, []).append((x[0] // 1024, x[1]))

    for key in data:
        data[key] = sorted(data[key], key=lambda x: x[0])

    _, combined = lineplot(data, 'Patch Representations', ylabel='Chamfer', xlabel='Size (KB)', width=8, height=6, legend=True)

    code = preamble + r'''
    \begin{document}
    %s
    \end{document}
    ''' % combined

    # combined = document_template % code
    # print('code', combined)
    synthesize_tex(code, 'geometry-images.pdf')

def frequencies(db):
    import torchvision

    from scipy.signal import savgol_filter

    losses_k4 = {}
    losses_k8 = {}
    losses_k16 = {}

    for k, d in db.items():
        k = 'L = ' + str(k)
        print('k = ', k)

        losses_k4[k] = []
        losses_k8[k] = []
        losses_k16[k] = []

        d0 = d['loss'][:250]
        d1 = d['loss'][250:500]
        d2 = d['loss'][500:]

        y0 = savgol_filter(d0, 25, 4)
        y1 = savgol_filter(d1, 25, 4)
        y2 = savgol_filter(d2, 25, 4)

        for i, l in enumerate(y0):
            losses_k4[k].append((i, l))

        for i, l in enumerate(y1):
            losses_k8[k].append((i, l))

        for i, l in enumerate(y2):
            losses_k16[k].append((i, l))

    print('losses_k4', losses_k4.keys())

    _, c0 = lineplot(losses_k4, 'Optimization Pt. 1 $(k = 4)$', ylabel=r'\textsc{Render}', xlabel='Iteration', width=7, height=6, mode='transparent')
    _, c1 = lineplot(losses_k8, 'Optimization Pt. 2 $(k = 8)$', ylabel='{}', xlabel='Iteration', width=7, height=6, mode='transparent')
    _, c2 = lineplot(losses_k16, 'Optimization Pt. 3 $(k = 16)$', ylabel='{}', xlabel='Iteration', width=7, height=6, mode='transparent', legend=True)

    # Load the images
    images = {
            'ref' : db[0]['images']['ref'],
            'f0'  : db[0]['images']['mesh'],
            'f1'  : db[1]['images']['mesh'],
            'f2'  : db[2]['images']['mesh'],
            'f4'  : db[4]['images']['mesh'],
            'f8'  : db[8]['images']['mesh'],
            'f16' : db[16]['images']['mesh'],
    }

    # Whitespace removal cropbox
    cbox = cropbox(images)

    cbox1 = 130, 230, 70, 170
    cbox2 = 30, 130, 470, 570

    # Extract and export the images
    directory = os.path.join('resources', 'figures', 'generated')
    abs_directory = os.path.abspath(directory)
    os.makedirs(abs_directory, exist_ok=True)
    for k, img in images.items():
        pimg = os.path.join(directory, k.replace(':', '-') + '.png')

        img = img.permute(2, 0, 1)
        img = img[ :, cbox[2] : cbox[3] + 1, cbox[0] : cbox[1] + 1]
        alpha = (img.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ img, alpha ], dim=0)

        # And the insets as well
        inset1 = img[ :, cbox1[2] : cbox1[3] + 1, cbox1[0] : cbox1[1] + 1]
        inset2 = img[ :, cbox2[2] : cbox2[3] + 1, cbox2[0] : cbox2[1] + 1]

        pimg = os.path.join(directory, k + '.png')
        pinset1 = os.path.join(directory, k.replace(':', '-') + '-inset1.png')
        pinset2 = os.path.join(directory, k.replace(':', '-') + '-inset2.png')

        torchvision.utils.save_image(inset1, pinset1)
        torchvision.utils.save_image(inset2, pinset2)
        torchvision.utils.save_image(img, pimg)

        box1 = torch.tensor([ cbox1[0], cbox1[2], cbox1[1], cbox1[3] ]).unsqueeze(0)
        box2 = torch.tensor([ cbox2[0], cbox2[2], cbox2[1], cbox2[3] ]).unsqueeze(0)

        img = torchvision.io.read_image(pimg)
        rgb = img[:3]
        rgb = torchvision.utils.draw_bounding_boxes(rgb, boxes=box1, colors="red", width=8)
        rgb = torchvision.utils.draw_bounding_boxes(rgb, boxes=box2, colors="blue", width=8)
        rgb = rgb / 255

        alpha = (rgb.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ rgb, alpha ], dim=0)
        torchvision.utils.save_image(img, pimg)

    combined = c0 + '\n' + c1 + '\n' + c2

    # TODO: preamble somewhere here
    code = preamble + r'''
    \begin{document}
        \setlength{\fboxsep}{0pt}
        \setlength{\fboxrule}{1pt}

        \centering

        \begin{minipage}{5cm}
            \centering
            \large
            \textsc{0 Fourier features}
            \vspace{1mm}
            \includegraphics[width=\textwidth]{%s}
        \end{minipage}
        \begin{minipage}{5cm}
            \centering
            \large
            \textsc{\textbf{8 Fourier features}}
            \vspace{1mm}
            \includegraphics[width=\textwidth]{%s}
        \end{minipage}
        \begin{minipage}{5cm}
            \centering
            \large
            \textsc{16 Fourier features}
            \vspace{1mm}
            \includegraphics[width=\textwidth]{%s}
        \end{minipage}
        \begin{minipage}{5cm}
            \centering
            \large
            \textsc{Reference}
            \vspace{1mm}
            \includegraphics[width=\textwidth]{%s}
        \end{minipage}

        \begin{tblr}{ colspec={ccccccc}, cells={valign=t, halign=c} }
                \sffamily
                0 FF & 1 FF & 2 FF & 4 FF & 8 FF & 16 FF & \textsc{Reference} \\
                \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}} \\
                \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}}
                & \fbox{\includegraphics[width=2cm]{%s}} \\
        \end{tblr}

        %s
    \end{document}''' % (
            abs_directory + '/f0.png',
            abs_directory + '/f8.png',
            abs_directory + '/f16.png',
            abs_directory + '/ref.png',

            abs_directory + '/f0-inset1.png',
            abs_directory + '/f1-inset1.png',
            abs_directory + '/f2-inset1.png',
            abs_directory + '/f4-inset1.png',
            abs_directory + '/f8-inset1.png',
            abs_directory + '/f16-inset1.png',
            abs_directory + '/ref-inset1.png',

            abs_directory + '/f0-inset2.png',
            abs_directory + '/f1-inset2.png',
            abs_directory + '/f2-inset2.png',
            abs_directory + '/f4-inset2.png',
            abs_directory + '/f8-inset2.png',
            abs_directory + '/f16-inset2.png',
            abs_directory + '/ref-inset2.png',

            combined
    )

    synthesize_tex(code, 'frequencies.pdf', log=True)

def losses(db):
    import torchvision

    from scipy.signal import savgol_filter

    data = {
            'Inverse Rendering': [],
            'Chamfer': []
    }

    for i, c in zip(db['time:ord'], db['loss:ord']):
        data['Inverse Rendering'].append((i, 1e6 * c))

    for i, c in zip(db['time:chm'], db['loss:chm']):
        data['Chamfer'].append((i, 1e6 * c))

    _, code = lineplot(data, '{}', ylabel=r'\textsc{Chamfer} $(\times 10^6)$', xlabel='Time (seconds)', width=8, height=8, mode='transparent', legend=True)

    # Load the images
    images = {
            'ref' : db['render:ref'],
            'ord' : db['render:ord'],
            'chm' : db['render:chm']
    }

    # Whitespace removal cropbox
    cbox = cropbox(images)

    # Extract and export the images
    directory = os.path.join('resources', 'figures', 'generated')
    abs_directory = os.path.abspath(directory)
    os.makedirs(abs_directory, exist_ok=True)
    for k, img in images.items():
        pimg = os.path.join(directory, k.replace(':', '-') + '.png')

        # img = (img/800).pow(1/2.2)
        img = img.permute(2, 0, 1)
        img = img[ :, cbox[2] : cbox[3] + 1, cbox[0] : cbox[1] + 1]
        alpha = (img.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ img, alpha ], dim=0)

        torchvision.utils.save_image(img, pimg)

    # TODO: preamble somewhere here
    code = preamble + r'''
    \begin{document}
        \setlength{\fboxsep}{0pt}
        \setlength{\fboxrule}{1pt}

        \centering

        \begin{minipage}{8cm}
            \centering
            %s
        \end{minipage}
        \begin{minipage}{5cm}
            \centering
            \large
            \textsc{Chamfer}
            \vspace{2mm}

            \includegraphics[width=\textwidth]{%s}
        \end{minipage}
        \begin{minipage}{5cm}
            \centering
            \large
            \textsc{\textbf{Inverse rendering}}
            \vspace{2mm}

            \includegraphics[width=\textwidth]{%s}
        \end{minipage}
        \begin{minipage}{5cm}
            \centering
            \large
            \textsc{Reference}
            \vspace{2mm}

            \includegraphics[width=\textwidth]{%s}
        \end{minipage}
    \end{document}''' % (
            code,
            abs_directory + '/chm.png',
            abs_directory + '/ord.png',
            abs_directory + '/ref.png'
    )

    synthesize_tex(code, 'loss-evaluation.pdf', log=True)

def ingp(db):
    import torchvision

    # Load the images
    images = {
        'ref': db['ref'],
        'ngf': db['ngf']['image'],
        'ngp11': db['ngp11']['image'],
        'ngp12': db['ngp12']['image'],
        'ngp13': db['ngp13']['image']
    }

    # Whitespace removal cropbox
    cbox = cropbox(images)

    # Extract and export the images
    directory = os.path.join('resources', 'figures', 'generated')
    abs_directory = os.path.abspath(directory)
    os.makedirs(abs_directory, exist_ok=True)

    cbox1 = 300, 400, 400, 550
    for k, img in images.items():
        img = img.permute(2, 0, 1)
        img = img[ :, cbox[2] : cbox[3] + 1, cbox[0] : cbox[1] + 1]
        alpha = (img.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ img, alpha ], dim=0)

        # And the insets as well
        inset1 = img[ :, cbox1[2] : cbox1[3] + 1, cbox1[0] : cbox1[1] + 1]

        pimg = os.path.join(directory, k + '.png')
        pinset1 = os.path.join(directory, k.replace(':', '-') + '-inset1.png')

        torchvision.utils.save_image(inset1, pinset1)
        torchvision.utils.save_image(img, pimg)

        box1 = torch.tensor([ cbox1[0], cbox1[2], cbox1[1], cbox1[3] ]).unsqueeze(0)

        img = torchvision.io.read_image(pimg)
        rgb = img[:3]
        rgb = torchvision.utils.draw_bounding_boxes(rgb, boxes=box1, colors="red", width=8)
        rgb = rgb / 255

        alpha = (rgb.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ rgb, alpha ], dim=0)
        torchvision.utils.save_image(img, pimg)

    # TODO: preamble somewhere here
    code = preamble + r'''
    \begin{document}
        \setlength{\fboxsep}{0pt}
        \setlength{\fboxrule}{1pt}

        \centering

        \begin{minipage}{20cm}
            \centering
            \begin{tblr}{ colspec={cc}, cells={valign=t, halign=c} }
                \textsc{Reference}
                & \textsc{NGF (Ours)} \\
                \fbox{\includegraphics[width=6cm]{%s}}
                & \fbox{\includegraphics[width=6cm]{%s}} \\
                Chamfer distance $\left(\times 10^6\right)$ & %.2f \\
            \end{tblr}
            \vspace{5mm}

            \begin{tblr}{ colspec={ccc}, cells={valign=t, halign=c} }
                \textsc{INGP \# 1}
                & \textsc{INGP \# 2}
                & \textsc{INGP \# 3} \\
                $(T = 2^{11}, F = 8)$
                & $(T = 2^{12}, F = 4)$
                & $(T = 2^{13}, F = 2)$ \\
                \fbox{\includegraphics[width=6cm]{%s}}
                & \fbox{\includegraphics[width=6cm]{%s}}
                & \fbox{\includegraphics[width=6cm]{%s}} \\
                %.2f & %.2f & %.2f \\
            \end{tblr}
        \end{minipage}
        \begin{minipage}{6cm}
            \begin{tblr}{ colspec={cc}, cells={valign=t, halign=c} }
                \textsc{NGF (Ours)} & \textsc{INGP \# 1} \\
                \fbox{\includegraphics[width=2.5cm]{%s}}
                & \fbox{\includegraphics[width=2.5cm]{%s}} \\
                \textsc{INGP \# 2} & \textsc{INGP \# 3} \\
                \fbox{\includegraphics[width=2.5cm]{%s}}
                & \fbox{\includegraphics[width=2.5cm]{%s}}
            \end{tblr}
        \end{minipage}
    \end{document}''' % (
            abs_directory + '/ref.png',
            abs_directory + '/ngf.png',
            1e6 * db['ngf']['error'],
            abs_directory + '/ngp11.png',
            abs_directory + '/ngp12.png',
            abs_directory + '/ngp13.png',
            1e6 * db['ngp11']['error'],
            1e6 * db['ngp12']['error'],
            1e6 * db['ngp13']['error'],
            abs_directory + '/ngf-inset1.png',
            abs_directory + '/ngp11-inset1.png',
            abs_directory + '/ngp12-inset1.png',
            abs_directory + '/ngp13-inset1.png',
    )

    synthesize_tex(code, 'ingp.pdf', log=True)


def teaser(db):
    import torchvision

    # Load the images
    images = {
        'ref': db['nrm:ref'],
        'ngf': db['nrm:ngf'],
        'qslim': db['nrm:qslim'],
        'nvdiff': db['nrm:nvdiff'],
        'ingp': db['nrm:ingp'],
        'patches': db['patches'],
    }

    # Whitespace removal cropbox
    cbox = cropbox(images)

    # Extract and export the images
    directory = os.path.join('resources', 'figures', 'generated')
    abs_directory = os.path.abspath(directory)
    os.makedirs(abs_directory, exist_ok=True)

    cbox1 = 100, 200, 000, 100
    cbox2 = 200, 300, 420, 520

    for k, img in images.items():
        img = img.permute(2, 0, 1)
        img = img[ :, cbox[2] : cbox[3] + 1, cbox[0] : cbox[1] + 1]
        alpha = (img.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ img, alpha ], dim=0)
        if k == 'patches':
            pimg = os.path.join(directory, k.replace(':', '-') + '.png')
            torchvision.utils.save_image(img, pimg)
            continue

        # And the insets as well
        inset1 = img[ :, cbox1[2] : cbox1[3] + 1, cbox1[0] : cbox1[1] + 1]
        inset2 = img[ :, cbox2[2] : cbox2[3] + 1, cbox2[0] : cbox2[1] + 1]

        pimg = os.path.join(directory, k.replace(':', '-') + '.png')
        pinset1 = os.path.join(directory, k.replace(':', '-') + '-inset1.png')
        pinset2 = os.path.join(directory, k.replace(':', '-') + '-inset2.png')

        torchvision.utils.save_image(inset1, pinset1)
        torchvision.utils.save_image(inset2, pinset2)
        torchvision.utils.save_image(img, pimg)

        box1 = torch.tensor([ cbox1[0], cbox1[2], cbox1[1], cbox1[3] ]).unsqueeze(0)
        box2 = torch.tensor([ cbox2[0], cbox2[2], cbox2[1], cbox2[3] ]).unsqueeze(0)

        img = torchvision.io.read_image(pimg)
        rgb = img[:3]
        rgb = torchvision.utils.draw_bounding_boxes(rgb, boxes=box1, colors="red", width=8)
        rgb = torchvision.utils.draw_bounding_boxes(rgb, boxes=box2, colors="blue", width=8)
        rgb = rgb / 255

        alpha = (rgb.sum(dim=0) > 0).unsqueeze(0)
        img = torch.concat([ rgb, alpha ], dim=0)
        torchvision.utils.save_image(img, pimg)

    # TODO: preamble somewhere here
    code = preamble + r'''
    \begin{document}
        \setlength{\fboxsep}{0pt}
        \setlength{\fboxrule}{2pt}

        \centering

        \begin{minipage}{10cm}
            \centering
            \includegraphics[width=\textwidth]{%s}
            \Large
            \textsc{Reference}

            15.5 MB
        \end{minipage}
        \begin{minipage}{5cm}
            \centering
            \includegraphics[width=\textwidth]{%s}

            \Large
            \textsc{NGF} (Patches)
        \end{minipage}
        \begin{minipage}{10cm}
            \centering
            \includegraphics[width=\textwidth]{%s}
            \Large
            \textsc{NGF (Ours)}

            324 KB
            \Large

            $50\times$ Compression
        \end{minipage}
        \begin{tblr}{ colspec={ccc}, cells={valign=t, halign=c} }
            \rotatebox{90}{\parbox{3cm}{\large \centering \textsc{QSlim} \\ %.2f}}
            &
            \renewcommand\fbox{\fcolorbox{red}{white}}
            \fbox{\includegraphics[width=3cm]{%s}}
            &
            \renewcommand\fbox{\fcolorbox{blue}{white}}
            \fbox{\includegraphics[width=3cm]{%s}} \\
            \rotatebox{90}{\parbox{3cm}{\large \centering \textsc{Nvdiffmodeling} \\ %.2f}}
            &
            \renewcommand\fbox{\fcolorbox{red}{white}}
            \fbox{\includegraphics[width=3cm]{%s}}
            &
            \renewcommand\fbox{\fcolorbox{blue}{white}}
            \fbox{\includegraphics[width=3cm]{%s}} \\
            \rotatebox{90}{\parbox{3cm}{\large \centering \textsc{Instant NGP} \\ %.2f}}
            &
            \renewcommand\fbox{\fcolorbox{red}{white}}
            \fbox{\includegraphics[width=3cm]{%s}}
            &
            \renewcommand\fbox{\fcolorbox{blue}{white}}
            \fbox{\includegraphics[width=3cm]{%s}} \\
            \rotatebox{90}{\parbox{3cm}{\large \centering \textsc{\textbf{NGF}} \\ \textbf{%.2f}}}
            &
            \renewcommand\fbox{\fcolorbox{red}{white}}
            \fbox{\includegraphics[width=3cm]{%s}}
            &
            \renewcommand\fbox{\fcolorbox{blue}{white}}
            \fbox{\includegraphics[width=3cm]{%s}} \\
            \rotatebox{90}{\parbox{3cm}{\large \centering \textsc{Reference} \\ Chamfer $\left(\times 10^6\right)$}}
            &
            \renewcommand\fbox{\fcolorbox{red}{white}}
            \fbox{\includegraphics[width=3cm]{%s}}
            &
            \renewcommand\fbox{\fcolorbox{blue}{white}}
            \fbox{\includegraphics[width=3cm]{%s}} \\
        \end{tblr}
    \end{document}''' % (
            abs_directory + '/ref.png',
            abs_directory + '/patches.png',
            abs_directory + '/ngf.png',

            1e6 * db['chamfer:qslim'],
            abs_directory + '/qslim-inset1.png',
            abs_directory + '/qslim-inset2.png',

            1e6 * db['chamfer:nvdiff'],
            abs_directory + '/nvdiff-inset1.png',
            abs_directory + '/nvdiff-inset2.png',

            1e6 * db['chamfer:ingp'],
            abs_directory + '/ingp-inset1.png',
            abs_directory + '/ingp-inset2.png',

            1e6 * db['chamfer:ngf'],
            abs_directory + '/ngf-inset1.png',
            abs_directory + '/ngf-inset2.png',

            abs_directory + '/ref-inset1.png',
            abs_directory + '/ref-inset2.png',
    )

    synthesize_tex(code, 'teaser.pdf', log=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='loss', help='Type of plot to generate')
    parser.add_argument('--db', type=str, default='', help='Database to plot from')
    parser.add_argument('--key', type=str, default='dpm', help='Key to plot from')
    parser.add_argument('--dir', type=str, default='.', help='Directory to plot from')
    parser.add_argument('--tick', type=int, default=5, help='Tick step for plots')
    parser.add_argument('--primary', type=str, help='Primary model to display')

    args = parser.parse_args()

    if args.type == 'loss':
        loss_plot(args.dir)
    elif args.type == 'results':
        db = torch.load(args.db)
        name = os.path.basename(args.db)
        name = name.split('.')[0]
        results_plot(name, db)
    elif args.type == 'table':
        dbs = {}
        for root, directory, files in os.walk(args.dir):
            if not root == os.path.basename(args.dir):
                continue

            for file in files:
                if file.endswith('.pt'):
                    db = torch.load(os.path.join(root, file))
                    f = file.split('.')[0].capitalize()
                    dbs[f] = db

        table(dbs)
    elif args.type == 'tessellation':
        db = torch.load(args.db)
        tessellation(db)
    elif args.type == 'features':
        db = torch.load(args.db)
        features(db)
    elif args.type == 'multichart':
        db = json.load(open(args.db))
        mutlichart(db)
    elif args.type == 'frequencies':
        db = torch.load(args.db)
        frequencies(db)
    elif args.type == 'losses':
        db = torch.load(args.db)
        losses(db)
    elif args.type == 'ingp':
        db = torch.load(args.db)
        ingp(db)
    elif args.type == 'display':
        db = torch.load(args.db)
        display(db)
    elif args.type == 'teaser':
        db = torch.load(args.db)
        teaser(db)
    else:
        raise NotImplementedError
