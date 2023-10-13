import os

models = [
    'pos-enc', 'onion-enc', 'morlet-enc', 'feat-enc'
    #'feat-morlet-enc', 'feat-onion-enc',
]

losses = [
    'v', 'n', 'vn', 'vnf'
    # 'v', 'n', 'vn', 'vn10', 'vn100', 'vnf'
]

lerps = [ 'linear', 'sin', 'floor', 'smooth-floor', ]

base = 'python3 multi.py --target meshes/rock_source_opt.pt --source meshes/rock_source.obj'
dir = 'results/rock/'
for m in models:
    # TODO: more exhaustic later
    # for l in losses:
    l = losses[2]
    for r in lerps:
        cmd = base + f' --model {m} --loss {l} --lerp {r} --output {dir}/{m}-{l}-{r} --iterations=10000'
        print(cmd)
        os.system(cmd)
