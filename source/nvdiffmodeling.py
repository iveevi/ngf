import os
import sys
import json
import subprocess
import shutil

assert len(sys.argv) == 3, 'Usage: python __file__ <reference> <directory with simplified models>'

PYTHON = sys.executable
SCRIPT = os.environ['HOME'] + '/sources/nvdiffmodeling/train.py'

# Make sure the reference mesh is a .obj file
ref = sys.argv[1]
assert ref.endswith('.obj'), 'Reference mesh must be a .obj file'
assert os.path.exists(ref),  'Reference mesh does not exist'

# Find all qslim-* models in the directory
models = []
# for f in os.listdir(sys.argv[2]):
#     if f.startswith('qslim-') and f.endswith('.obj'):
#         models.append(f)

for root, dirs, files in os.walk(sys.argv[2]):
    for f in files:
        if f.startswith('qslim-') and f.endswith('.obj'):
            models.append(os.path.join(root, f))

print('Found {} models'.format(len(models)))
print(models)

# Create JSON configs for each model and run the training
name = os.path.basename(sys.argv[2])
print('Running nvdiffmodeling on {} models'.format(name))
tmp = 'results/comparisons/nvdiffmodeling/' + name
os.makedirs(tmp, exist_ok=True)

for m in models:
    base = os.path.basename(m)
    base = os.path.splitext(base)[0]
    data = {
            'base_mesh': m,
            'ref_mesh': ref,
            'camera_eye': [ 2.5, 0.0, -2.5 ],
            'camera_up': [ 0.0, 1.0, 0.0 ],
            'random_textures': True,
            'iter': 5000,
            'save_interval': 250,
            'train_res': 512,
            'batch': 8,
            'learning_rate': 0.001,
            'min_roughness' : 0.25,
            'out_dir' : os.path.join(tmp, base)
    }

    print('Running', data)
    config = os.path.join(tmp, base + '.json')
    with open(config, 'w') as f:
        json.dump(data, f, indent=4)

    cmd = '{} {} --config {}'.format(PYTHON, SCRIPT, config)
    subprocess.run(cmd.split())

    # Copy the resulting mesh and put elsewhere
    nvdiff = os.path.join(data['out_dir'], 'mesh/mesh.obj')
    assert os.path.exists(nvdiff), 'nvdiffmodeling did not produce a mesh'

    # Get the resolution from the qslim object
    out = m.replace('qslim', 'nvdiffmodeling')
    # out = os.path.join(sys.argv[2], out)
    print('Copying {} to {}'.format(nvdiff, out))
    shutil.copy(nvdiff, out)
