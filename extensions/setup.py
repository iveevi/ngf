from setuptools import setup
from torch.utils import cpp_extension

# TODO: split the sources, and into separate submodules
setup(name = 'optext',
    ext_modules = [
        cpp_extension.CppExtension('optext',[ 'optext.cu' ], extra_compile_args=['-O3']),
    ],
    include_dirs = [ '..', '../thirdparty/glm' ],
    cmdclass = { 'build_ext': cpp_extension.BuildExtension },
    version='0.1.0',
    description='Geometric optimization extension')
