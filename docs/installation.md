# Installation

Use a separate environment. The original recipe is Python 3.10, PyTorch 2.2.2/CUDA
12.1, MKL 2024.0.0, NumPy 1.26.4 and Coal 3.0.1. A compatible C++ compiler, Eigen
headers and CUDA toolkit with `nvcc` are required. The clean-host recipe below still
needs independent validation; local validation uses an isolated Conda clone with
private editable installs removed and extensions rebuilt from this checkout.

```bash
conda create -n hugs-bodex python=3.10 -y
conda activate hugs-bodex
conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install coal==3.0.1 eigen -c conda-forge -y
conda install pytorch-scatter -c pyg -y
python -m pip install --force-reinstall mkl==2024.0.0 numpy==1.26.4
python -m pip install hydra-core pybind11 ninja
git submodule update --init --recursive
python -m pip install -e . --no-build-isolation
(cd src/curobo/geom/cpp && python -m pip install . --no-build-isolation)
```

CUDA extensions build through `setup.py`; the Coal OpenMP wrapper uses
`src/curobo/geom/cpp/coal_parallel.cpp`. `CONDA_PREFIX` must name the active environment
for the Coal build. Configure `TORCH_CUDA_ARCH_LIST` and `MAX_JOBS` for the target GPU
and build resources. No prebuilt extension is included in source.

Viewer dependencies:

```bash
python -m pip install -e third_party/pytorch_kinematics -e third_party/utils_python
python -m pip install viser
```

Render additionally needs `pyrender`, `imageio`, `pyglet<2` and EGL. Prior viewing needs
compatible public `manopth` (https://github.com/hassony2/manopth), `chumpy`, and separately
licensed MANO models via `task.mano_root`; its real-data/dependency gate is pending.
No USD/pxr package is required or offered.

```bash
python -c 'import curobo,pytorch_kinematics,mr_utils; print(curobo.__file__,pytorch_kinematics.__file__,mr_utils.__file__)'
python -c 'import importlib.util; assert importlib.util.find_spec("pxr") is None'
```

All packages must resolve inside this checkout/environment. Dependencies are pinned
by submodule commits, not floating branch tips.
