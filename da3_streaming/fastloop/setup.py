from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Note: Eigen should be in /usr/include/eigen3
setup(
    name='sim3solve',
    ext_modules=[
        CppExtension(
            name='sim3solve',
            sources=['solve.cpp'],
            include_dirs=['/usr/include/eigen3'],
            extra_compile_args=['-O3'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
