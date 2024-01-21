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

# LaTeX lineplot preset
# TODO: modularize this
document_template = r'''
\documentclass[varwidth=500cm, border=0pt]{standalone}

\renewcommand{\familydefault}{\sfdefault}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{libertine}
\usepackage{pgfplots}
\usepackage{tikz}
\usepackage{siunitx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}

\usetikzlibrary{calc}

\definecolor{color0}{HTML}{ea7aa5}
\definecolor{color1}{HTML}{65d59e}
\definecolor{color2}{HTML}{ab92e6}
\definecolor{color3}{HTML}{b2c65d}
\definecolor{color4}{HTML}{e19354}

\pgfplotsset{compat=1.18}
\pgfplotsset{
    yticklabel style={
        /pgf/number format/fixed,
        /pgf/number format/precision=5
    },
    scaled y ticks=false
}

\begin{document}
\begin{center}
%s
\end{center}
\end{document}
'''

combined_template = r'''
\documentclass[varwidth=500cm, border=0pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{libertine}measure
\usepackage{graphicx}

\begin{document}
\begin{center}
\begin{figure}
    \centering
    \includegraphics[width=15cm]{%s}

    \includegraphics[width=15cm]{%s}
\end{figure}
\end{center}
\end{document}
'''

# y label style={at={(1.07, 0.5)}, rotate=180},
# legend style={draw=none, fill=none, at={(0.5, -0.4)}, anchor=north, font=\large},
lineplot_template = r'''\begin{tikzpicture}[baseline=(current bounding box.north)]
\begin{axis}[
    ymode=log,
    title style={font=\large},
    title=%s,
    name=%s, %s
    xshift=2cm,
    width=%.2f cm,
    height=%.2f cm,
    ylabel=%s,
    xlabel=%s,
    legend columns=%d,
    legend style={fill=none, font=\scriptsize},
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
                                kwargs['xlabel'], 1,
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
    line = '\\addplot [line width=1pt, color=%s, opacity=0.65] coordinates {' % color

    for v in d:
        line += '(%.4f, %.4f) ' % v
    line += '};\n'
    line += '\\label{' + key + '}'
    if legend:
        line += r'\addlegendentry{\large \textsc{' + key.capitalize() + '}}\n'

    # Add coordinate at the end
    line += '\\addplot [mark=*, color=%s, forget plot] coordinates { (%f, %f) };' % (color, d[-1][0], d[-1][1])

    return line

def lineplot(data, name, xlabel='X', ylabel='', xtick_step=10, width=8, height=6, mode='marked', at=None, legend=False):
    codename = name.lower().replace(' ', '-')
    title = '\\textsc{%s}' % name #.capitalize()
    print('Codename', codename, 'at', at)

    # Generate LaTeX code
    colors = [ 'color0', 'color1', 'color2', 'color3', 'color4' ]

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

    # Find best tick step (5, 10, 25, etc.)
    # tick_step = (x_max - x_min)
    # tick_step_10 = 10 ** np.floor(np.log10(tick_step))
    # tick_step_5 = tick_step_10 / 2
    # tick_step = tick_step_10 if tick_step / tick_step_10 > 25 else tick_step_5
    # tick_step = int(tick_step)

    loc = '' if at is None else 'at={(%s)},' % at
    addplot = '\n'.join(lines)

    tex = fill_lineplot(title=title, codename=codename, loc=loc, width=width,
                        height=height, xlabel=xlabel, ylabel=ylabel,
                        num_cols=len(data), x_min=x_min, x_max=x_max,
                        y_min=y_min, y_max=y_max, tick_step=args.tick,
                        plots=addplot)

    return codename, tex

def synthesize_tex(code, filename):
    import tempfile
    import subprocess

    # Write to temporary file
    with tempfile.NamedTemporaryFile('w+') as fp:
        print('Writing to', fp.name)
        fp.write(code)
        fp.seek(0)

        os.makedirs(os.path.dirname('media/figures/generated/'), exist_ok=True)
        subprocess.check_call(['xelatex', '-interaction=nonstopmode', fp.name],
            cwd=os.path.dirname('media/figures/generated/')) #, stdout=subprocess.DEVNULL)

        # Copy resulting PDF to destination
        result = os.path.join('media/figures/generated/', os.path.basename(fp.name) + '.pdf')
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
        if mind > abs(entry['count'] - 2500):
            mind = abs(entry['count'] - 2500)
            primary = i

    directory = os.path.join('media', 'figures', 'generated')
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
        'armadillo' : (350, 450, 100, 200),
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
        rgb = torchvision.utils.draw_bounding_boxes(rgb, boxes=box, colors="red", width=8) / 255
        img = torch.concat([ rgb, img[None, 3] ], dim=0)
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

    # # Generate lineplots
    _, code0 = lineplot(render_data, '', ylabel=r'\textsc{Render}', xlabel='{}', width=6, height=4)
    _, code2 = lineplot(chamfer_data, '', ylabel=r'\textsc{Chamfer}', xlabel='Compression ratio', width=6, height=4, legend=True)

    combined = code0 + '\n' + code2

    # Create the code
    round_ten = lambda x: 5 * ((x + 4) // 5)

    code = r'''
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

    \definecolor{color0}{HTML}{ea7aa5}
    \definecolor{color1}{HTML}{65d59e}
    \definecolor{color2}{HTML}{ab92e6}
    \definecolor{color3}{HTML}{b2c65d}
    \definecolor{color4}{HTML}{e19354}

    \pgfplotsset{compat=1.18}
    \pgfplotsset{
        yticklabel style={
            /pgf/number format/fixed,
            /pgf/number format/precision=5
        },
        scaled y ticks=false
    }

    %%\newcolumntype{b}{>{\columncolor{blue!25}}c}

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

    synthesize_tex(code, os.path.join('media', 'figures', name + '.pdf'))

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

        data = converted[t]

        sizem = [ d['size'] for d in list(data.values()) ]
        sizem = int(sum(sizem)/len(sizem))

        row = []
        for f in field_names:
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

    directory     = os.path.join('media', 'figures', 'generated')
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

    _, render_code = lineplot(render, '', xlabel='Tessellation', ylabel='Render', width=6, height=5)
    _, normal_code = lineplot(normal, '', xlabel='Tessellation', ylabel='Normal', width=6, height=5)
    _, chamfer_code = lineplot(chamfer, '', xlabel='Tessellation', ylabel='Chamfer', width=10, height=5, legend=True)

    combined = render_code + '\n' + normal_code + '\n\n' + chamfer_code

    code = r'''
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

    \definecolor{color0}{HTML}{ea7aa5}
    \definecolor{color1}{HTML}{65d59e}
    \definecolor{color2}{HTML}{ab92e6}
    \definecolor{color3}{HTML}{b2c65d}
    \definecolor{color4}{HTML}{e19354}

    \pgfplotsset{compat=1.18}
    \pgfplotsset{
        yticklabel style={
            /pgf/number format/fixed,
            /pgf/number format/precision=5
        },
        scaled y ticks=false
    }

    \captionsetup[subfigure]{justification=centering}

    \begin{document}
        \setlength{\fboxsep}{0pt}
        \setlength{\fboxrule}{1pt}

        \begin{tblr}{ colspec={cccc}, cells={valign=t, halign=c} }
            \begin{minipage}{3cm}
                \centering
                \textsc{Reference}\vspace{1mm}
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
                \textsc{Resolution %d}\vspace{1mm}
                \includegraphics[width=\textwidth]{%s}
            \end{minipage}
            & \\
        \end{tblr}
    \end{document}''' % (
        abs_directory + '/ref.png',
        2, abs_directory + '/tess2.png',
        4, abs_directory + '/tess4.png',
        combined,
        8, abs_directory + '/tess8.png',
        12, abs_directory + '/tess12.png',
        16, abs_directory + '/tess16.png'
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

    directory     = os.path.join('media', 'figures', 'generated')
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

        img = torch.concat([ rgb, img[None, 3] ], dim=0)
        torchvision.utils.save_image(img, pimg)

    print(args.primary, primary.keys())

    code = r'''
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

    \definecolor{color0}{HTML}{ea7aa5}
    \definecolor{color1}{HTML}{65d59e}
    \definecolor{color2}{HTML}{ab92e6}
    \definecolor{color3}{HTML}{b2c65d}
    \definecolor{color4}{HTML}{e19354}

    \pgfplotsset{compat=1.18}
    \pgfplotsset{
        yticklabel style={
            /pgf/number format/fixed,
            /pgf/number format/precision=5
        },
        scaled y ticks=false
    }

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

    _, code = lineplot(data, 'Patch Representations', ylabel='Chamfer', xlabel='Size (KB)', width=10, height=6, xtick_step=50, legend=True)

    combined = document_template % code
    print('code', combined)
    synthesize_tex(combined, 'geometry-images.pdf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='loss', help='Type of plot to generate')
    parser.add_argument('--db', type=str, default='results.json', help='Database to plot from')
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
    else:
        raise NotImplementedError
