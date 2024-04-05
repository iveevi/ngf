from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [
    'cluster.cpp',
    'mesh.cpp',
    'ngfutil.cu',
    'parametrize.cpp',
    'smoothing.cu',
    'triangulate.cu',
]

setup(
    name='ngfutil',
    ext_modules=[
        CUDAExtension(
            name='ngfutil',
            sources=sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-ccbin=/usr/bin/g++-12',
                    '-Xcudafe', '--diag_suppress=20011',
                    '-Xcudafe', '--diag_suppress=20014',
                ]
            },
            libraries=['assimp'])
    ],
    include_dirs=['..', '../thirdparty/glm'],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='0.1.0',
    description='Utility kernels for optimizing neural geometry fields')
