import json
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import sys

# TODO: pass catags with argparse
assert len(sys.argv) >= 3, 'Usage: python plot.py <database> <key>'

db = sys.argv[1]
key = sys.argv[2]
patterns = sys.argv[3:]

regexes = []
for pattern in patterns:
    regexes.append(re.compile(pattern))

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

    sns.set()
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    # Plot dpm vs size
    # TODO: white list certain catags...
    for catag, resolutions in second_level_parents:
        if not any(regex.match(catag) for regex in regexes):
            continue

        l = []
        for k, v in resolutions.items():
            dpm = v['dpm']
            size = v['size']/1024
            # l.append((size, -np.log10(dpm)))
            l.append((size, dpm))

        l.sort(key=lambda x: x[0])
        x, y = zip(*l)
        ax[0].plot(x, y, label=catag, marker='o')

    ax[0].set_ylabel('-log10(dpm)')
    ax[0].set_xlabel('size (KB)')
    ax[0].set_xscale('log')
    ax[0].legend()

    # Plot the atomic elements (dnormal vs size)
    for catag, resolutions in second_level_parents:
        if not any(regex.match(catag) for regex in regexes):
            continue

        l = []
        for k, v in resolutions.items():
            dnormal = v['dnormal']
            size = v['size']/1024
            # l.append((size, -np.log10(dnormal)))
            l.append((size, dnormal))

        l.sort(key=lambda x: x[0])
        x, y = zip(*l)
        ax[1].plot(x, y, label=catag, marker='o')

    ax[1].set_ylabel('-log10(dnormal)')
    ax[1].set_xlabel('size (KB)')
    ax[1].set_xscale('log')
    ax[1].legend()

    # Plot the atomic elements (render vs size)
    for catag, resolutions in second_level_parents:
        if not any(regex.match(catag) for regex in regexes):
            continue

        l = []
        for k, v in resolutions.items():
            render = v['render']
            size = v['size']/1024
            l.append((size, -np.log10(render)))
            # l.append((size, render))

        l.sort(key=lambda x: x[0])
        x, y = zip(*l)
        ax[2].plot(x, y, label=catag, marker='o')

    ax[2].set_ylabel('render')
    ax[2].set_xlabel('size (KB)')
    ax[2].set_xscale('log')
    ax[2].legend()

    plt.show()
