import os
import torch
import numpy as np

from configurations import *
from util import *

if __name__ == '__main__':
    # Create a directory for all the compiled models
    os.makedirs('compiled', exist_ok=True)

    # Load all directories in the results
    results = os.listdir('results')

    # Iterate over all directories
    for result in results:
        # Get all .pt files in the directory
        files = os.listdir('results/' + result)
        files = [ file for file in files if file.endswith('.pt') ]
        print('Found {} files in {}'.format(len(files), result))

        for file in files:
            # Load the model
            model = torch.load('results/{}/{}'.format(result, file))
            print('\nLoaded model from {}'.format(file))

            mlp = model['model']

            make = mlp.__class__.__name__
            if not 'Onion' in make:
                print('Model is not onion based, skipping...')
                continue

            kernel = model['kernel']
            if not kernel == 'linear':
                print('Kernel is not linear, skipping...')
                continue

            # Destination path
            path = 'compiled/{}/{}'.format(result, file).replace('.pt', '.nsc')
            print('Compiling model to {}'.format(path))

            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                s0, s1, s2 = mlp.s0.item(), mlp.s1.item(), mlp.s2.item()
                ss = np.array([ s0, s1, s2 ], dtype='float32')
                print('Scales: {}, {}, {}'.format(s0, s1, s2))

                weights = [ L.weight.data for L in mlp.encoding_linears.cpu() ]
                biases  = [ L.bias.data   for L in mlp.encoding_linears.cpu() ]

                print('Weights: {}'.format([ w.shape for w in weights ]))
                print('Biases:  {}'.format([ b.shape for b in biases ]))

                complexes = model['complexes'].detach().cpu()
                points    = model['points'].detach().cpu()
                features  = model['features'].detach().cpu()

                print(model.keys())
                print('Complexes: {}'.format(complexes.shape))
                print('Points:    {}'.format(points.shape))
                print('Features:  {}'.format(features.shape))

                sizes = np.array([ complexes.shape[0], points.shape[0], features.shape[1] ], dtype='int32')
                assert features.shape[0] == points.shape[0]
                print('Sizes: {}'.format(sizes))

                complexes_bytes = complexes.numpy().astype('int32').tobytes()
                points_bytes    = points.numpy().astype('float32').tobytes()
                features_bytes  = features.numpy().astype('float32').tobytes()

                weights_bytes = b''.join([ w.numpy().astype('float32').tobytes() for w in weights ])
                biases_bytes  = b''.join([ b.numpy().astype('float32').tobytes() for b in biases ])

                # Writing data to file
                f.write(ss.tobytes())
                f.write(sizes.tobytes())

                f.write(complexes_bytes)
                f.write(points_bytes)
                f.write(features_bytes)

                for w in weights:
                    f.write(w.shape[0].to_bytes(4, 'little'))
                    f.write(w.shape[1].to_bytes(4, 'little'))
                    f.write(w.numpy().astype('float32').tobytes())

                for b in biases:
                    f.write(b.shape[0].to_bytes(4, 'little'))
                    f.write(b.numpy().astype('float32').tobytes())

                f.close()

                # Evaluate for reference
                kernel = model['kernel']
                kernel = lerps[ kernel ]

                lp, lf = sample(model['complexes'], model['points'], model['features'], 3, kernel)
                vs = model['model'].cuda()(points=lp, features=lf)
