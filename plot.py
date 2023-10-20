import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

# TODO: pass catags with argparse
assert len(sys.argv) == 3, 'Usage: python plot.py <database> <key>'

db = sys.argv[1]
key = sys.argv[2]

# def get_direct_atomic_parents(data):
#     if not isinstance(data, dict):
#         assert False, 'Empty data'
#     else:
#         direct_parent = True
#         for k, v in data.items():
#             if isinstance(v, dict):
#                 direct_parent = False
#                 break
#
#         if direct_parent:
#             return data
#         else:
#             l = []
#             for k, v in data.items():
#                 lv = get_direct_atomic_parents(v)
#                 print(lv)
#                 if isinstance(lv, list):
#                     l.extend(lv)
#                 else:
#                     l.append(lv)
#             return l

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
    # print(data)

    # Get all the atomic elements (lowest level dicts)
    # atomic_parents = get_direct_atomic_parents(data)
    # print(atomic_parents)

    second_level_parents = get_secondary_atomic_parents(data)
    print(second_level_parents)

    sns.set()
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plot dpm vs size
    # TODO: white list certain catags...
    for catag, resolutions in second_level_parents:
        l = []
        for k, v in resolutions.items():
            dpm = v['dpm']
            size = v['size']
            l.append((size, -np.log10(dpm)))

        l.sort(key=lambda x: x[0])
        x, y = zip(*l)
        ax[0].plot(x, y, label=catag, marker='o')

    ax[0].set_ylabel('-log10(dpm)')
    ax[0].set_xlabel('size')
    ax[0].set_xscale('log')
    ax[0].legend()

    # Plot the atomic elements (dnormal vs size)
    for catag, resolutions in second_level_parents:
        l = []
        for k, v in resolutions.items():
            dnormal = v['dnormal']
            size = v['size']
            l.append((size, -np.log10(dnormal)))

        l.sort(key=lambda x: x[0])
        x, y = zip(*l)
        ax[1].plot(x, y, label=catag, marker='o')

    ax[1].set_ylabel('-log10(dnormal)')
    ax[1].set_xlabel('size')
    ax[1].set_xscale('log')
    ax[1].legend()

    plt.show()
