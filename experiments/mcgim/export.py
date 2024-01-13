import os
import re
import torch

pattern = re.compile('^mcgim*.pt$')
for root, _, files in os.walk('results/nefertiti'):
    for file in files:
        if pattern.match(file):
            mcgim = os.path.join(root, file)
            mcgim = torch.load(mcgim)

