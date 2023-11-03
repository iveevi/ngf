import os
import subprocess
import sys
import json

# TODO: multi GPU support

CONFIGS = os.path.join(os.path.dirname(__file__), 'configs')
PROGRAM = os.path.join(os.path.dirname(__file__), 'combined.py')
PYTHON = sys.executable

# Find all config files
configs = []
for f in os.listdir(CONFIGS):
    if f.endswith('.json'):
        configs.append(os.path.join(CONFIGS, f))

# Process all configs
commands = []

for config in sorted(configs):
    with open(config, 'r') as f:
        data = json.load(f)

    print('Detected config file', config)
    confirm = input('  Run? [y/n] ')
    if confirm.lower() != 'y':
        continue

    model = data['name']
    target = data['target']
    directory = data['directory']

    print('  > Model name:       ', model)
    print('  > Target:           ', target)
    print('  > Results directory:', directory, end='\n\n')

    # Build all commands to run
    for experiment in data['experiments']:
        name = experiment['name']
        source = experiment['source']
        method = experiment['method']
        clusters = experiment['clusters']
        batch = experiment['batch']
        resolution = experiment['resolution']

        print('  > Experiment:       ', name)
        print('    > Source:         ', source)
        print('    > Method:         ', method.split('/'))
        print('    > Clusters:       ', clusters)
        print('    > Batch size:     ', batch)
        print('    > Resolution:     ', resolution)

        cmd = f'{PYTHON} {PROGRAM} {model} {target} {source} {method} {clusters} {batch} {resolution} {directory}'
        print('    > Command:        ', cmd)
        commands.append(cmd)
