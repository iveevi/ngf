import subprocess
import sys
import os

assert len(sys.argv) == 3, 'Usage: python simplify.py <model> <directory>'

binary = './build/simplify'
rates = [ 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90 ]

for rate in rates:
    result = os.path.join(sys.argv[2], f'qslim-r{rate:03d}.obj')
    cmd = subprocess.list2cmdline([binary, sys.argv[1], result, str(rate/100.0)])
    subprocess.call(cmd, shell=True)
