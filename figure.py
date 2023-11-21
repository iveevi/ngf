import argparse
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Global parameters
plt.rcParams['text.usetex'] = True

# Different types of plots
# 1. Loss plots (gather from directory)
def loss_plot(dir):
    # Gather all .losses files
    files = [ f for f in os.listdir(dir) if f.endswith('.losses') ]
    print(files)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # ax.set_prop_cycle(sns.color_palette('Spectral'))
    # ax.set_prop_cycle(sns.color_palette('Paired'))

    # Plot each file
    for f in files:
        # Get name
        name = f.split('.')[0]

        # Get data
        data = []
        with open(os.path.join(dir, f), 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                data = line.split(',')
                data = [ float(d) for d in data ]

        # Plot data
        p = ax.plot(data, label=name, alpha=0.5)
        ax.scatter(len(data) - 1, data[-1], color=p[0].get_color())
        ax.set_yscale('log')

    # Add legend
    ax.legend()

    # Save figure
    # fig.savefig(os.path.join(dir, 'losses.pdf'), bbox_inches='tight')
    fig.savefig('losses.pdf', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.', help='Directory to plot from')
    parser.add_argument('--type', type=str, default='loss', help='Type of plot to generate')

    args = parser.parse_args()

    if args.type == 'loss':
        loss_plot(args.dir)

