import json
import os
import re
import subprocess
import sys

# TODO: multi GPU support

CONFIGS = os.path.join(os.path.dirname(__file__), '../configs')
PROGRAM = os.path.join(os.path.dirname(__file__), '../combined.py')
PYTHON = sys.executable

# If arguments are provided, then compile the regex
patterns = None
if len(sys.argv) > 1:
    patterns = [ re.compile(p) for p in sys.argv[1:] ]

# Find all config files
configs = []
for f in os.listdir(CONFIGS):
    if f.endswith('.json'):
        configs.append(os.path.join(CONFIGS, f))

# Process all configs
commands = []

for config in sorted(configs):
    # If patterns are provided, then check if the config file matches
    if patterns is not None:
        match = False
        for pattern in patterns:
            if pattern.search(config):
                match = True
                break

        if not match:
            continue

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
        fixed = experiment['fixed']

        assert type(fixed) == bool

        print('  > Experiment:       ', name)
        print('    > Source:         ', source)
        print('    > Method:         ', method.split('/'))
        print('    > Clusters:       ', clusters)
        print('    > Batch size:     ', batch)
        print('    > Resolution:     ', resolution)
        print('    > Fixed:          ', fixed)

        # TODO: also mark name with source object LOD
        result = os.path.join(directory, name + '.pt')
        print('    > Result:         ', result)

        cmd = f'{PYTHON} {PROGRAM} {target} {source} {method} {clusters} {batch} {resolution} {result} {fixed}'
        print('    > Command:        ', cmd, end='\n\n')
        commands.append(cmd)

for cmd in commands:
    print('  > ', cmd)
    subprocess.run(cmd, shell=True)
