import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re
import seaborn as sns
import sys

# TODO: pass catags with argparse
assert len(sys.argv) >= 3, 'Usage: python plot.py <database> <key>'

db       = sys.argv[1]
key      = sys.argv[2]
patterns = sys.argv[3:]

regexes = []
for pattern in patterns:
    regexes.append(re.compile(pattern))

# TeX formatting
sns.set_theme(style="whitegrid")
palette = sns.color_palette("hls", 8)

plt.rcParams['text.usetex'] = True
def textsc(str):
    return r'\textsc{' + str + '}'

def refmt(str):
    splits = str.split('-')
    for i in range(len(splits)):
        splits[i] = splits[i].capitalize()
        # if it ends with a number, add as subscript
        if splits[i][-1].isdigit():
            splits[i] = splits[i][:-1] + '$_{' + splits[i][-1] + '}$'

    return ' '.join(splits)

def get_secondary_atomic_parents(data, catag=''):
    if not isinstance(data, dict):
        assert False, 'Empty data'
    else:
        direct_parent = True
        for k, v in data.items():
            if isinstance(v, dict):
                direct_parent = False
                break

        if direct_parent:
            return -1

        second_level = True
        for k, v in data.items():
            if get_secondary_atomic_parents(v) != -1:
                second_level = False
                break

        if second_level:
            return (catag, data)
        else:
            l = []
            for k, v in data.items():
                ncatag = catag + ('' if catag == '' else '-') + k
                lv = get_secondary_atomic_parents(v, ncatag)
                if isinstance(lv, list):
                    l.extend(lv)
                else:
                    l.append(lv)
            return l

with open(db) as f:
    data = json.load(f)
    data = data[key]

    print('Plotting {}...'.format(key))

    second_level_parents = get_secondary_atomic_parents(data)
    print(second_level_parents)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')

    # fig.suptitle(refmt(key + '-metrics'))

    # Plot dpm vs size
    # TODO: sort by catag
    max_cratio = 0
    for catag, resolutions in second_level_parents:
        if not any(regex.match(catag) for regex in regexes):
            continue

        l = []
        for k, v in resolutions.items():
            normal = v['normal']
            # size = v['size']/1024
            cratio = v['cratio']
            l.append((cratio, normal))

            max_cratio = max(max_cratio, cratio)

        l.sort(key=lambda x: x[0])
        x, y = zip(*l)
        ax[0].plot(x, y, label=(refmt(catag)), marker='o')

    # ax[0].set_title('Normal vs Size')
    # ax[0].set_xlabel('Compression Ratio')
    ax[0].set_ylabel('Normal Loss')
    # ax[0].set_xlabel('size (KB)')
    # ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    # ax[0].legend()

    ax[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax[0].ticklabel_format(style='plain', axis='x')
    # ax[0].legend(loc='lower right', fancybox=True, framealpha=1, shadow=True, borderpad=1, markerscale=0)

    # Plot the atomic elements (render vs size)
    for catag, resolutions in second_level_parents:
        if not any(regex.match(catag) for regex in regexes):
            continue

        l = []
        for k, v in resolutions.items():
            render = v['render']
            # size = v['size']/1024
            cratio = v['cratio']
            l.append((cratio, render))

        l.sort(key=lambda x: x[0])
        x, y = zip(*l)
        ax[1].plot(x, y, label=(refmt(catag)), marker='o')

    # ax[1].set_title('Render vs Size')
    # ax[1].set_xlabel('Compression Ratio')
    ax[1].set_ylabel('Render Loss')
    # ax[1].set_xlabel('size (KB)')
    # ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    # ax[1].legend()

    ax[1].yaxis.set_minor_formatter(ticker.FormatStrFormatter('%.2f'))
    ax[1].ticklabel_format(style='plain', axis='x')
    # ax[1].legend(loc='lower right', fancybox=True, framealpha=1, shadow=True, borderpad=1, markerscale=0)

    # Plot the atomic elements (chamfer vs size)
    for catag, resolutions in second_level_parents:
        if not any(regex.match(catag) for regex in regexes):
            continue

        l = []
        for k, v in resolutions.items():
            chamfer = v['chamfer']
            # size = v['size']/1024
            cratio = v['cratio']
            l.append((cratio, chamfer))

        l.sort(key=lambda x: x[0])
        x, y = zip(*l)
        ax[2].plot(x, y, label=(refmt(catag)), marker='o')

    # ax[2].set_title('Chamfer vs Size')
    ax[2].set_xlabel('Compression Ratio')
    ax[2].set_ylabel('Chamfer Distance')
    # ax[2].set_xlabel('Size (KB)')
    # ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    # ax[2].legend()

    ax[2].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax[2].ticklabel_format(style='plain', axis='x')
    # ax[2].legend(loc='lower right', fancybox=True, framealpha=1, shadow=True, borderpad=1, markerscale=0)

    # plt.xlim(0, None)

    # Collect all the labels to put in the figure legend
    handles, labels = ax[0].get_legend_handles_labels()

    # handles2, labels2 = ax[1].get_legend_handles_labels()
    # handles3, labels3 = ax[2].get_legend_handles_labels()
    # handles.extend(handles2)
    # handles.extend(handles3)
    # labels.extend(labels2)
    # labels.extend(labels3)

    # Plot the legend
    # TODO: also alter the color palette
    fig.legend(handles, labels, loc='outside upper center', ncol=len(handles), fancybox=True, framealpha=1, shadow=True, borderpad=1, markerscale=0, fontsize='small')

    # fig.tight_layout()

    plt.savefig(os.path.join('figures', key + '_metrics.pdf'), format='pdf')
    plt.show()
