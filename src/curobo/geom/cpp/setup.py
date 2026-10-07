import os
from pathlib import Path

import pybind11
from setuptools import Extension, setup


def _find_prefix():
    """Find the native dependency prefix used by Conda or cmeel/uv."""
    if os.environ.get("CMEEL_PREFIX"):
        return Path(os.environ["CMEEL_PREFIX"])

    try:
        import cmeel_pth

        return Path(cmeel_pth.__file__).parent / cmeel_pth.CMEEL_PREFIX
    except (ImportError, AttributeError):
        if os.environ.get("CONDA_PREFIX"):
            return Path(os.environ["CONDA_PREFIX"])
        raise RuntimeError(
            "Could not find Coal's native dependency prefix. Activate the Conda environment "
            "or install coal with uv before building the extension."
        ) from None


# Coal installed from PyPI uses cmeel.prefix inside the uv environment. Conda
# installations continue to use CONDA_PREFIX when cmeel is not installed. Prefer
# the active Python's cmeel prefix over a parent shell's stale CONDA_PREFIX.
prefix = _find_prefix()
lib_dir = prefix / "lib"
include_dir = prefix / "include"
eigen_include_dir = Path(os.environ.get("EIGEN3_INCLUDE_DIR", include_dir / "eigen3"))
if not (eigen_include_dir / "Eigen" / "Core").is_file():
    raise RuntimeError(
        f"Eigen headers not found in {eigen_include_dir}. Run uv sync to install "
        "cmeel-eigen, or set EIGEN3_INCLUDE_DIR to an existing Eigen include directory."
    )

# Define the extension module
extension_mod = Extension(
    "coal_openmp_wrapper",
    sources=["coal_parallel.cpp"],
    library_dirs=[str(lib_dir)],
    libraries=[
        "coal",
        "boost_filesystem",
        "qhull_r",
        "octomap",
        "octomath",
        "assimp",
        "stdc++",
        "gcc_s",
        "pthread",
        "m",
        "rt",
        "c",
    ],
    include_dirs=[str(include_dir), str(eigen_include_dir), pybind11.get_include()],
    # extra_compile_args=["-fopenmp", "-std=c++11"],  # OpenMP and C++11 support
    extra_compile_args=["-fopenmp", "-std=c++14"],  # OpenMP and C++11 support
    extra_link_args=[f"-L{lib_dir}", "-lcoal", "-fopenmp"],
    runtime_library_dirs=[str(lib_dir)],
    language="c++",
)

# Setup function
setup(name="coal_openmp_wrapper", version="0.1", ext_modules=[extension_mod], install_requires=["pybind11"])
