import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

module_root = Path(__file__).parent.absolute()
csrc = module_root / 'faster_beta_splatting' / 'csrc'

cxx_flags = ['/std:c++17' if os.name == 'nt' else '-std=c++17', '-O3' if os.name != 'nt' else '/O2']
nvcc_flags = ['-std=c++17', '-O3', '-use_fast_math']

setup(
    name='faster_beta_splatting',
    author='Hierarchical Incremental Beta Splatting, Linus Franke (derived from Faster-GS by Florian Hahlbohm, Apache-2.0)',
    description='Fused-Adam beta-kernel rasterizer for online incremental reconstruction.',
    packages=['faster_beta_splatting'],
    ext_modules=[
        CUDAExtension(
            name='faster_beta_splatting._C',
            sources=[str(csrc / 'rasterizer.cu')],
            include_dirs=[str(csrc)],
            extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags},
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
