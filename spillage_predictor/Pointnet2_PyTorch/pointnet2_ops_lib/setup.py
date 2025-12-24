import os
import glob
import os.path as osp
from setuptools import setup, find_packages

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

requirements = ["torch>=1.4"]

exec(open(osp.join("pointnet2_ops", "_version.py")).read())

os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"

# Delay torch imports completely inside this function
def get_extensions():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    return [
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
        )
    ], {"build_ext": BuildExtension}

# Get extensions and cmdclass after torch is installed
ext_modules, cmdclass = get_extensions()

setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
)

