import os
import argparse
import configparser
import subprocess

from pynvml import *

# Parse arguments
parser = argparse.ArgumentParser(description='Configure the experiment.')
parser.add_argument('-c', '--config', type=str, default='config.ini', help='Path to the configuration file.')
parser.add_argument('-e', '--experiments', nargs='+', type=str, default=[ 'all' ], help='Experiments to run.')
args = parser.parse_args()

if not os.path.isfile(args.config):
    print('[!] Error: Configuration file does not exist.')
    exit()

config = configparser.ConfigParser()
config.read(args.config)

experiments = []
for section in config.sections():
    if section in args.experiments or 'all' in args.experiments:
        experiments.append(section)

print(experiments)

# Initialize GPU
nvmlInit()

# Run experiments
for experiment in experiments:
    print('\nRunning experiment: {}'.format(experiment))

    # iterations = config[experiment]['iterations']
    # print('  Iterations: {}'.format(iterations))

    reference = config[experiment]['reference']
    print('  Reference: {}'.format(reference))

    quads = config[experiment]['quads']
    print('  Quads: {}'.format(quads))

    results = config[experiment]['results']
    print('  Directory: {}'.format(results))

    cmd = 'python train.py -q ' + quads + ' -r ' + reference + ' -d ' + results
    print('  Command: {}'.format(cmd))

    print('available memory: ', nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).free)

    # TODO: pool processes and check for available resources (memory)
    # TODO: sbuprocess, instantiate as many as possible
    # os.system(cmd)

    subprocess.Popen(cmd, shell=True)
