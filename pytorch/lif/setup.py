import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')

setup(
    name='lif',
    version='0.1',
    author='fzr95',
    author_email='fzr95@outlook.com',
    description='lif for snn',
    long_description='lif for snn',
    ext_modules=[
        CppExtension(
            name='lif',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)