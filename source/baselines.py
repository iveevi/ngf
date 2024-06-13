import os
import sys
import json
import shutil
import trimesh
import argparse
import fast_simplification

from tqdm import trange

from ngf import NGF

def qslim_do(base, reduction):
    vout, fout = fast_simplification.simplify(base.vertices, base.faces, reduction)
    sizekb = (vout.nbytes + fout.nbytes) / 1024
    return vout, fout, sizekb

def qslim_search(base, sizekb):
    small, large = 0.0005, 0.9995
    for _ in trange(15, ncols=50, desc='QSlim', leave=False):
        mid = (small + large) / 2
        vout, fout, qsizekb = qslim_do(base, mid)
        if qsizekb < sizekb:
            large = mid
        else:
            small = mid
    return vout, fout

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', type=str, nargs='+')
    parser.add_argument('--reference', type=str)
    parser.add_argument('--directory', type=str)

    args = parser.parse_args(sys.argv[1:])

    reference = trimesh.load(args.reference)

    for file in args.pt:
        basename = os.path.basename(file)
        basename = basename.split('.')[0]

        sizekb = os.path.getsize(file) / 1024
        print(f'\nPROCESSING {basename} [{sizekb:.2f} KB]')

        # Copy as an OBJ
        local_ref = os.path.basename(args.reference)
        local_ref = local_ref.split('.')[0] + '.obj'
        local_ref = os.path.join(args.directory, local_ref)

        m = trimesh.Trimesh(reference.vertices, reference.faces)
        m.vertex_normals
        m.export(local_ref)

        # QSlim
        vout, fout = qslim_search(reference, sizekb)
        qslim_destination = basename + '-qslim.stl'
        qslim_sizekb = (vout.nbytes + fout.nbytes) / 1024
        
        qslim_destination = os.path.join(args.directory, qslim_destination)
        qslim_mesh = trimesh.Trimesh(vout, fout)
        qslim_mesh.export(qslim_destination)
        
        qslim_alt_destination = basename + '-qslim.obj'
        qslim_alt_destination = os.path.join(args.directory, qslim_alt_destination)
        qslim_mesh.export(qslim_alt_destination)
        
        print(f'    EXPORTED QSLIM RESULT AS {qslim_destination} [{qslim_sizekb:.2f} KB]')

        # nvdiffmodeling
        nvdiffmodeling_destination = os.path.join(args.directory, basename + '-nvdiffmodeling')

        config = {
                'base_mesh': qslim_alt_destination,
                'ref_mesh': local_ref,
                'random_textures': False,
                'iter': 1000,
                'save_interval': 250,
                'train_res': 512,
                'batch': 5,
                'learning_rate': 1e-4,
                'out_dir' : nvdiffmodeling_destination
        }

        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            json.dump(config, open(tmp.name, 'w'), indent=4)

            DIRECTORY = os.path.dirname(__file__)
            PYTHON = sys.executable
            SCRIPT = os.path.abspath(DIRECTORY + '/../thirdparty/nvdiffmodeling/train.py')

            cmd = '{} {} --config {}'.format(PYTHON, SCRIPT, tmp.name)
            os.system(cmd)
        
            print(f'    EXPORTED NVDIFFMODELING RESULT AS {nvdiffmodeling_destination} [{qslim_sizekb:.2f} KB]')

        m = trimesh.load(os.path.join(nvdiffmodeling_destination, 'mesh', 'mesh.obj'))
        nvdiffmodeling_destination = basename + '-nvdiffmodeling.stl'
        nvdiffmodeling_destination = os.path.join(args.directory, nvdiffmodeling_destination)
        m.export(nvdiffmodeling_destination)
