import argparse
import figuregen
import json
import matplotlib.pyplot as plt
import os
import simpleimageio as sio
import numpy as np

from figuregen.util.image import Cropbox, relative_mse
from figuregen.util.templates import CropComparison

# LaTeX lineplot preset
# TODO: modularize this
document_template = r'''
\documentclass[varwidth=500cm, border=0pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{libertine}
\usepackage{pgfplots}
\usepackage{tikz}
\usepackage{siunitx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}

\definecolor{color0}{HTML}{ea7aa5}
\definecolor{color1}{HTML}{65d59e}
\definecolor{color2}{HTML}{ab92e6}
\definecolor{color3}{HTML}{b2c65d}
\definecolor{color4}{HTML}{e19354}

\pgfplotsset{compat=1.16}
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
\usepackage{libertine}
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

lineplot_template = r'''\begin{tikzpicture}
\begin{axis}[
    name=%s,%s
    xshift=2cm,
    width=%.2f cm,
    height=%.2f cm,
    ylabel=\textsc{%s},
    xlabel=\textsc{%s},
    legend style={at={(0.5, 1.02)}, anchor=south},
    legend columns=%d,
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
    return lineplot_template % (kwargs['codename'], kwargs['loc'],
                                kwargs['width'], kwargs['height'],
                                kwargs['ylabel'], kwargs['xlabel'],
                                kwargs['num_cols'], kwargs['x_min'],
                                kwargs['x_max'], kwargs['y_min'],
                                kwargs['y_max'], kwargs['tick_step'],
                                kwargs['plots'])

def plot_marked(key, d, color, legend):
    line = '\\addplot [mark=*, color=%s] coordinates {' % color

    for v in d:
        line += '(%.4f, %.4f) ' % v
    line += '};'

    if legend:
        line += '\\addlegendentry{\\textsc{' + key + '}}'

    return line

def plot_transparent_end(key, d, color, legend):
    line = '\\addplot [line width=1pt, color=%s, opacity=0.65] coordinates {' % color

    for v in d:
        line += '(%.4f, %.4f) ' % v
    line += '};'

    if legend:
        line += '\\addlegendentry{\\textsc{' + key + '}}\n'

    # Add coordinate at the end
    line += '\\addplot [mark=*, color=%s, forget plot] coordinates { (%.4f, %.4f) };' % (color, d[-1][0], d[-1][1])

    return line

def lineplot(data, name, xlabel='X', ylabel='Y', width=8, height=6, at=None, legend=False):
    codename = name.lower().replace(' ', '-')
    print('Codename', codename, 'at', at)

    # Generate LaTeX code
    colors = [ 'color0', 'color1', 'color2', 'color3', 'color4' ]

    lines = []
    for color, (key, d) in zip(colors, data.items()):
        # lines.append(plot_marked(key, d, color, legend))
        lines.append(plot_transparent_end(key, d, color, legend))

    x_min = min([ min([ v[0] for v in d ]) for d in data.values() ])
    x_max = max([ max([ v[0] for v in d ]) for d in data.values() ])
    stride = (x_max - x_min) / 25
    x_min = max(0, x_min - stride)
    x_max += stride

    y_min = min([ min([ v[1] for v in d ]) for d in data.values() ])
    y_max = max([ max([ v[1] for v in d ]) for d in data.values() ])
    stride = (y_max - y_min) / 25
    y_min = max(0, y_min - stride)
    y_max += stride

    # Find best tick step (5, 10, 25, etc.)
    tick_step = (x_max - x_min)
    tick_step_10 = 10 ** np.floor(np.log10(tick_step))
    tick_step_5 = tick_step_10 / 2
    tick_step = tick_step_10 if tick_step / tick_step_10 > 25 else tick_step_5
    tick_step = int(tick_step)
    print('Tick step', tick_step)

    print('X min/max', x_min, x_max)
    print('Y min/max', y_min, y_max)

    loc = '' if at is None else 'at={(%s)},' % at
    addplot = '\n'.join(lines)

    tex = fill_lineplot(codename=codename, loc=loc, width=width, height=height,
                        xlabel=xlabel, ylabel=ylabel, num_cols=len(data),
                        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                        tick_step=tick_step, plots=addplot)

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
            cwd=os.path.dirname('media/figures/generated/'), stdout=subprocess.DEVNULL)

        # Copy resulting PDF to destination
        result = os.path.join('media/figures/generated/', os.path.basename(fp.name) + '.pdf')
        subprocess.check_call(['cp', result, filename])
        print('Retrieving', result, 'into', filename)

# Batch crop
def cropbox(images):
    boxes = []
    for image in images.values():
        rgb = image[:,:,:3]
        sums = np.sum(rgb, axis=2)
        mask = np.where(sums < 3, 1, 0)

        indices = np.where(mask == 1)
        rgb_box = (np.min(indices[1]), np.min(indices[0]), np.max(indices[1]), np.max(indices[0]))

        boxes.append(rgb_box)

    left = min([ box[0] for box in boxes ])
    top = min([ box[1] for box in boxes ])
    right = max([ box[2] for box in boxes ])
    bottom = max([ box[3] for box in boxes ])

    print('Common box', left, top, right, bottom)

    return Cropbox(top=top, left=left, height=bottom - top, width=right - left, scale=1)

# Results plots (gather from directory)
def results_plot(name, db):
    # TODO: custom tex generator for the figure
    # also remove the whitespace...
    # TODO: use exr images...
    # TODO: tonemapping...
    # TODO: align the normal map (rotate) so that it is blue

    # Load all images from the database and crop them to relavant regions
    images = {}
    for method in db.values():
        for entry in method:
            for img in entry['images'].values():
                images[img] = sio.read(img)

    # print('Images', images.keys())

    display = cropbox(images)
    for img in images.keys():
        images[img] = display.crop(images[img])

        # Tone map
        images[img] = np.power(images[img], 1/2.2)
        images[img] = np.clip(images[img], 0, 1)

    # Cropbox for all scenes
    cropboxes = {
        'armadillo'     : Cropbox(top=75, left=450, height=200, width=250),
        'chinese'       : Cropbox(top=250, left=1000, height=200, width=200, scale=5),
        'dragon-statue' : Cropbox(top=0, left=200, height=300, width=300, scale=5),
        'lucy'          : Cropbox(top=0, left=200, height=100, width=150),
        'nefertiti'     : Cropbox(top=0, left=200, height=150, width=200),
        'planck'        : Cropbox(top=200, left=120, height=150, width=200),
        'roal'          : Cropbox(top=250, left=150, height=200, width=200),
    }

    cbox = cropboxes[name]

    # Unpack database
    keys = list(db.keys())
    print('Keys: {}'.format(keys))

    # Render subfigure
    def textsc(text):
        return r'\textsc{' + text + '}'

    ref_grid = figuregen.Grid(num_rows=2, num_cols=2)
    inset_grid = figuregen.Grid(num_rows=2, num_cols=1 + len(keys))

    target_render = db[keys[0]][0]['images']['render-target']
    target_normal = db[keys[0]][0]['images']['normal-target']

    target_render = images[target_render]
    target_normal = images[target_normal]

    # Set reference full
    e = ref_grid.get_element(0, 0)
    e.set_image(figuregen.PNG(target_render))
    e.set_marker(cbox.get_marker_pos(), cbox.get_marker_size(), color=[255, 0, 0])

    e = ref_grid.get_element(1, 0)
    e.set_image(figuregen.PNG(target_normal))
    e.set_marker(cbox.get_marker_pos(), cbox.get_marker_size(), color=[255, 0, 0])

    ref_grid.set_row_titles(txt_list=[ 'Render', 'Normal' ], position='left')
    ref_grid.set_col_titles(txt_list=[ textsc('Reference'), textsc('Ours') ], position='top')

    # Set reference inset
    target_inset_render = cbox.crop(target_render)
    target_inset_normal = cbox.crop(target_normal)

    e = inset_grid.get_element(0, 0)
    e.set_image(figuregen.PNG(target_inset_render))
    e.set_frame(linewidth=1, color=[0, 0, 0])
    # e.set_caption(r'$\mathcal{L}_ 1$')

    e = inset_grid.get_element(1, 0)
    e.set_image(figuregen.PNG(target_inset_normal))
    e.set_frame(linewidth=1, color=[0, 0, 0])
    # e.set_caption(r'$\mathcal{L}_ 1$')

    cols = [ 'Reference' ] + [ key for key in keys ]
    inset_grid.set_col_titles(txt_list=[ textsc(key) for key in cols ], position='top')

    for i, key in enumerate(keys):
        img = db[key][0]['images']['render-source']
        img = images[img]

        if key == 'NGF':
            grid_render = ref_grid.get_element(0, 1)
            grid_render.set_image(figuregen.PNG(img))
            grid_render.set_marker(cbox.get_marker_pos(), cbox.get_marker_size(), color=[255, 0, 0])

        # TODO: PSNR as well...
        e = inset_grid.get_element(0, 1 + i)
        l1 = np.abs(target_render - img).mean()
        e.set_caption('{:.3f}'.format(l1))
        e.set_frame(linewidth=1, color=[0, 0, 0])

        render_inset = cbox.crop(img)
        e.set_image(figuregen.PNG(render_inset))

        img = db[key][0]['images']['normal-source']
        img = images[img]

        if key == 'NGF':
            grid_render = ref_grid.get_element(1, 1)
            grid_render.set_image(figuregen.PNG(img))
            grid_render.set_marker(cbox.get_marker_pos(), cbox.get_marker_size(), color=[255, 0, 0])

        e = inset_grid.get_element(1, 1 + i)
        l1 = np.abs(target_normal - img).mean()
        e.set_caption('{:.3f}'.format(l1))
        e.set_frame(linewidth=1, color=[0, 0, 0])

        normal_inset = cbox.crop(img)
        e.set_image(figuregen.PNG(normal_inset))

    lay = ref_grid.get_layout()
    lay.set_padding(right=1)

    os.makedirs(os.path.dirname('media/figures/generated/'), exist_ok=True)
    figuregen.horizontal_figure([ ref_grid, inset_grid ], width_cm=15, filename=os.path.join('media/figures/generated', name + '-insets.pdf'))

    # Collect lineplot data
    render_data = {}
    normal_data = {}
    chamfer_data = {}

    for key in keys:
        render_data[key] = []
        normal_data[key] = []
        chamfer_data[key] = []

        for entry in db[key]:
            cratio = entry['cratio']
            render_data[key].append((cratio, entry['render']))
            normal_data[key].append((cratio, entry['normal']))
            chamfer_data[key].append((cratio, entry['chamfer']))

        # Sort by cratio
        render_data[key] = sorted(render_data[key], key=lambda x: x[0])
        normal_data[key] = sorted(normal_data[key], key=lambda x: x[0])
        chamfer_data[key] = sorted(chamfer_data[key], key=lambda x: x[0])

    print('Render data', render_data)
    print('Normal data', normal_data)
    print('Chamfer data', chamfer_data)

    # Generate lineplots
    cn0, code0 = lineplot(render_data, 'Render Loss')
    cn1, code1 = lineplot(normal_data, 'Normal Loss', at=cn0 + '.south east', legend=True)
    _,   code2 = lineplot(chamfer_data, 'Chamfer Loss', at=cn1 + '.south east')

    combined = code0 + '\n' + code1 + '\n' + code2
    combined = document_template % combined

    synthesize_tex(combined, os.path.join('media/figures/generated', name + '-losses.pdf'))

    # Create combined figure
    path0 = os.path.join('media/figures/generated', name + '-insets.pdf')
    path1 = os.path.join('media/figures/generated', name + '-losses.pdf')

    path0 = os.path.abspath(path0)
    path1 = os.path.abspath(path1)

    combined = combined_template % (path0, path1)
    print('Combined', combined)

    synthesize_tex(combined, os.path.join('media/figures', name + '.pdf'))

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
    pointed = lambda f: [ (i, float(l)) for i, l in enumerate(csv(f)) ]

    graphs = { strip(f) : pointed(f) for f in files }

    _, tex = lineplot(graphs, 'Loss', ylabel='Loss', xlabel='Iterations',
                      width=12, height=7, legend=True)

    tex = document_template % tex
    synthesize_tex(tex, 'losses.pdf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='loss', help='Type of plot to generate')
    parser.add_argument('--db', type=str, default='results.json', help='Database to plot from')
    parser.add_argument('--key', type=str, default='dpm', help='Key to plot from')
    parser.add_argument('--dir', type=str, default='.', help='Directory to plot from')

    args = parser.parse_args()

    if args.type == 'loss':
        loss_plot(args.dir)
    elif args.type == 'results':
        db = json.load(open(args.db))
        if args.key == 'all':
            for key in db.keys():
                results_plot(key, db[key])
        else:
            results_plot(args.key, db[args.key])
    else:
        raise NotImplementedError
