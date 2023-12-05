import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import re
import seaborn as sns
import sys
import torch
import torchmetrics
import torchvision

from PIL import Image

# TODO: pass catags with argparse
assert len(sys.argv) >= 3, 'Usage: python plot.py <database> <key>'

db       = sys.argv[1]
name     = sys.argv[2]
patterns = sys.argv[3:]

regexes = []
for pattern in patterns:
    regexes.append(re.compile(pattern))

# TeX formatting
sns.set_theme(style="whitegrid")
palette = sns.color_palette("hls", 8)

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300

def itoa(x):
    if x >= 1000:
        return '{:.1f}K'.format(x / 1000)
    else:
        return str(x)

with open(db) as f:
    data = json.load(f)
    data = data[name]

    # Create a table
    fig = plt.figure(figsize=(20, 6), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, wspace=0.05, hspace=0.05, width_ratios=[2.5, 1])

    img_ax = subfigs[0].subplots(2, 4)
    ax     = subfigs[1].subplots(3, 1, sharex=True)

    # Plot dpm vs size
    max_cratio = 0
    for k, key in enumerate(data):
        complexes = data[key]

        chamfer  = []
        counts   = []
        cratios  = []
        dnormals = []
        dpms     = []
        normals  = []
        renders  = []
        src_imgs = []
        src_nrms = []
        tgt_imgs = []
        tgt_nrms = []

        complexes.sort(key=lambda x: x['cratio'])
        for entry in complexes:
            chamfer.append(entry['chamfer'])
            counts.append(entry['count'])
            cratios.append(entry['cratio'])
            dnormals.append(entry['dnormal'])
            dpms.append(entry['dpm'])
            normals.append(entry['normal'])
            renders.append(entry['render'])
            src_imgs.append(entry['images']['render-source'])
            src_nrms.append(entry['images']['normal-source'])
            tgt_imgs.append(entry['images']['render-target'])
            tgt_nrms.append(entry['images']['normal-target'])

        ax[0].plot(cratios, normals, label=key, marker='o')
        ax[1].plot(cratios, renders, label=key, marker='o')
        ax[2].plot(cratios, chamfer, label=key, marker='o')
        # ax[3].plot(cratios, dnormals, label=key, marker='o')
        # ax[4].plot(cratios, dpms, label=key, marker='o')

        color0 = ax[0].get_lines()[-1].get_color()
        color1 = ax[1].get_lines()[-1].get_color()
        color2 = ax[2].get_lines()[-1].get_color()
        # color3 = ax[3].get_lines()[-1].get_color()
        # color4 = ax[4].get_lines()[-1].get_color()

        # # Label each dot with the count
        # for i, count in enumerate(counts):
        #     ax[0].annotate(itoa(count), (cratios[i], normals[i]),
        #         # xytext=(-30, 0), textcoords='offset points',
        #         bbox=dict(boxstyle='round', fc=color0, alpha=0.5))
        #
        #     ax[1].annotate(itoa(count), (cratios[i], renders[i]),
        #         # xytext=(-30, 0), textcoords='offset points',
        #         bbox=dict(boxstyle='round', fc=color1, alpha=0.5))
        #
        #     ax[2].annotate(itoa(count), (cratios[i], chamfer[i]),
        #         # xytext=(0, 0), textcoords='offset points',
        #         bbox=dict(boxstyle='round', fc=color2, alpha=0.5))
        #
        #     # ax[3].annotate(itoa(count), (cratios[i], dnormals[i]),
        #     #     xytext=(-30, 0), textcoords='offset points',
        #     #     bbox=dict(boxstyle='round', fc=color3, alpha=0.5))
        #
        #     # ax[4].annotate(count, (cratios[i], dpms[i]),
        #     #     xytext=(-30, 0), textcoords='offset points',
        #     #     bbox=dict(boxstyle='round', fc=color4, alpha=0.5))

        # Error images
        t_img = Image.open(tgt_imgs[1])
        s_img = Image.open(src_imgs[1])

        t_nrm = Image.open(tgt_nrms[1])
        s_nrm = Image.open(src_nrms[1])

        t_img = torchvision.transforms.ToTensor()(t_img).permute(1, 2, 0)
        s_img = torchvision.transforms.ToTensor()(s_img).permute(1, 2, 0)

        t_nrm = torchvision.transforms.ToTensor()(t_nrm).permute(1, 2, 0)
        s_nrm = torchvision.transforms.ToTensor()(s_nrm).permute(1, 2, 0)

        e_img = (t_img - s_img).abs().mean(dim=2)
        e_nrm = (t_nrm - s_nrm).abs().mean(dim=2)

        psnr = torchmetrics.PeakSignalNoiseRatio()
        psnr_img = psnr(t_img, s_img)
        psnr_nrm = psnr(t_nrm, s_nrm)

        l1_img = e_img.mean()
        l1_nrm = e_nrm.mean()

        # print(key, psnr_img, psnr_nrm)

        # Preview images
        capped_key = key.upper()
        img_ax[0][k + 1].imshow(s_img)
        img_ax[0][k + 1].grid(False)
        img_ax[0][k + 1].set_xticks([])
        img_ax[0][k + 1].set_yticks([])
        img_ax[0][k + 1].set_xlabel(f'{{ {psnr_img:.2f}/{l1_img:.3f} }}')
        img_ax[0][k + 1].set_title(f'{{ {capped_key} }}')

        # Add the error image as an inset
        axins = img_ax[0][k + 1].inset_axes([0, 0, 0.35, 0.35])
        axins.imshow(e_img, cmap='coolwarm')
        axins.set_xticks([])
        axins.set_yticks([])

        img_ax[1][k + 1].imshow(s_nrm)
        img_ax[1][k + 1].grid(False)
        img_ax[1][k + 1].set_xticks([])
        img_ax[1][k + 1].set_yticks([])
        img_ax[1][k + 1].set_xlabel(f'{{ {psnr_nrm:.2f}/{l1_nrm:.3f} }}')

        # Add the error image as an inset
        axins = img_ax[1][k + 1].inset_axes([0, 0, 0.35, 0.35])
        axins.imshow(e_nrm, cmap='coolwarm')
        axins.set_xticks([])
        axins.set_yticks([])

        if k == 0:
            img_ax[0][0].imshow(t_img)
            img_ax[0][0].set_ylabel('Render')
            img_ax[0][0].grid(False)
            img_ax[0][0].set_xticks([])
            img_ax[0][0].set_yticks([])
            img_ax[0][0].set_title('{{ Ground Truth }}')

            img_ax[1][0].imshow(t_nrm)
            img_ax[1][0].set_ylabel('Normal')
            img_ax[1][0].grid(False)
            img_ax[1][0].set_xticks([])
            img_ax[1][0].set_yticks([])

    ax[0].set_ylabel('Normals')
    ax[1].set_ylabel('Renders')
    ax[2].set_ylabel('Chamfer')
    # ax[3].set_ylabel('DNormals')
    # ax[4].set_ylabel('DPM')

    ax[2].set_xlabel('Compression Ratio')

    for axis in ax:
        axis.yaxis.set_major_formatter(ticker.ScalarFormatter())
        axis.ticklabel_format(style='plain', axis='x')

    # Plot the legend
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, labels,
        loc='upper left',
        ncol=len(handles),
        fancybox=True,
        framealpha=0.5,
        shadow=False,
        borderpad=1,
        markerscale=0,
        fontsize='small')

    plt.savefig(os.path.join('media/figures', name + '_metrics.pdf'), format='pdf')
    plt.show()
