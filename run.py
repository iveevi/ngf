import subprocess
import argparse

cmds = [
        'python combined.py meshes/nefertiti pos linear 200',
        'python combined.py meshes/nefertiti onion linear 200',
        
        'python combined.py meshes/chinese pos linear 200',
        'python combined.py meshes/chinese onion linear 200',
        
        'python combined.py meshes/lucy pos linear 200',
        'python combined.py meshes/lucy onion linear 200',
        
        'python combined.py meshes/armadillo pos linear 200',
        'python combined.py meshes/armadillo onion linear 200',
]

for cmd in cmds:
    print('  >', cmd)
    subprocess.call(cmd.split())

# from configurations import *
#
# parser = argparse.ArgumentParser(description='Train models')
# parser.add_argument('--name', type=str, default='nefertiti', help='name of the experiment')
# parser.add_argument('--iterations', type=int, default=100_000, help='number of iterations')
# parser.add_argument('--gpus', type=int, default=2, help='number of gpus')
#
# args = parser.parse_args()
#
# gpu_processes = {}
#
# next_gpu = 0
# for m in models:
#     for l in lerps:
#         cmd = f'CUDA_VISIBLE_DEVICES={next_gpu} python train.py meshes/{args.name} {m} {l} {args.iterations}'
#         gpu_processes.setdefault(next_gpu, []).append(cmd)
#         next_gpu = (next_gpu + 1) % args.gpus
#
# # Fork for as many gpus as we have
# import os
#
# for gpu, cmds in gpu_processes.items():
#     pid = os.fork()
#     if pid == 0:
#         print(f'GPU {gpu} running {len(cmds)} commands')
#         for cmd in cmds:
#             print('  >', cmd)
#             os.system(cmd)
#         exit(0)
#     else:
#         print('Forked', pid)
