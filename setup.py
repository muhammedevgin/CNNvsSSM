from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='selective_scan_cpp',
    ext_modules=[
        cpp_extension.CppExtension(
            name='selective_scan_cpp',
            sources=['selective_scan.cpp'],
            extra_compile_args=['-O3', '-fopenmp', '-march=native'],
            extra_link_args=['-fopenmp']
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
