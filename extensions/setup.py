from setuptools import setup
from torch.utils import cpp_extension

# TODO: split the sources, and into separate submodules
setup(name = 'optext',
    ext_modules = [
        cpp_extension.CppExtension('optext', [
            'optext.cu',
            'cluster.cpp',
            'parametrize.cpp',
            'smoothing.cu',
            'triangulate.cu',
            'mesh.cpp',
        # ], extra_compile_args=['-O3'], libraries=['assimp']),
        ], extra_compile_args=['-g'], libraries=['assimp']),
    ],
    include_dirs = [ '..', '../thirdparty/glm' ],
    cmdclass = { 'build_ext': cpp_extension.BuildExtension },
    version='0.1.0',
    description='Geometric optimization extension')
