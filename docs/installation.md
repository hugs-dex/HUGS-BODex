# Installation

## Requirements

- Linux x86-64 with [uv](https://docs.astral.sh/uv/), Git, and a C++ compiler with
  OpenMP support (GCC/G++ 11 recommended).
- An NVIDIA GPU and a driver compatible with CUDA 12.1.
- A CUDA 12.x toolkit with `nvcc` on `PATH`. The toolkit is needed to compile
  extensions; installing PyTorch alone does not provide it.

This installation was tested on Ubuntu 22.04 with GCC 11.4, CUDA toolkit 12.4,
and an RTX 4090.

The installation below uses Python 3.10, PyTorch 2.2.2 with CUDA 12.1, NumPy 1.26.4,
Coal 3.0.1, and Eigen 3.4. After cloning, run commands from the repository root.
If `uv` is not installed, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Environment and dependencies

If needed, clone `https://github.com/hugs-dex/HUGS-BODex.git` and enter the
`HUGS-BODex` directory first.

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate

git submodule update --init --recursive

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS=4
uv pip install pip setuptools wheel
uv pip install --no-deps --no-build-isolation chumpy==0.70
uv sync --frozen --extra render --extra human
```

`uv sync` installs the dependencies recorded in `pyproject.toml` and `uv.lock`,
including the local `pytorch_kinematics` and `utils_python` submodules plus the
human-prior visualization packages. The virtual environment is stored in
`HUGS-BODex/.venv`; reactivate it in each new shell with `source .venv/bin/activate`.

chumpy 0.70 imports pip during its build without declaring it as a build dependency.
The separate installation above supplies the build tools and disables isolation only
for chumpy. `--no-deps` leaves its runtime dependencies to the locked `uv sync`;
chumpy remains part of the `human` extra. Repeat these steps when recreating `.venv`.

Eigen 3.4 headers are installed by `cmeel-eigen` inside `.venv` during `uv sync`.
No system Eigen installation or `sudo` is needed. The Coal wrapper build finds them
automatically under `.venv/lib/python3.10/site-packages/cmeel.prefix/include/eigen3`.

## Build and install

Coal's Python wheel bundles its native libraries under the uv environment, but the
separate OpenMP wrapper still needs to be compiled. Install its pinned native packages
after `uv sync`:

```bash
uv pip install --no-deps \
  coal==3.0.1 cmeel==0.61.0 cmeel-assimp==5.4.3.1 \
  cmeel-boost==1.87.0.1 cmeel-octomap==1.10.0 cmeel-qhull==8.0.2.1 \
  cmeel-zlib==1.3.2 \
  eigenpy==3.10.3
```

The `--no-deps` flag is intentional: the cmeel metadata currently requires NumPy 2.x,
while this project and `pytorch_kinematics` use NumPy 1.26.4. The required cmeel
runtime packages are pinned explicitly above.

For later synchronization, use `uv sync --frozen --inexact --extra render --extra human`
to preserve the separately installed Coal packages, wrapper, and `manopth`.

```bash
(cd src/curobo/geom/cpp && uv pip install . --no-build-isolation)
```

Keep `.venv` activated when building the Coal extension. The build script discovers
Coal's native headers and libraries from the uv environment automatically. By default,
the CUDA build targets visible GPUs. For a machine without a visible GPU, set
`TORCH_CUDA_ARCH_LIST` to the target architecture before building, for example `8.9`
for an RTX 4090. Reduce `MAX_JOBS` if compilation runs out of memory.

## Optional: human-prior visualization

Human-initialized synthesis only needs exported prior files. To also display the
human hand meshes in the prior viewer, install the pinned `manopth` Git dependency
(the other Python packages are included by the default `uv sync` command):

```bash
uv pip install --no-build-isolation \
  'manopth @ git+https://github.com/DexGrasp-TH/manopth.git@dd83a157dccd4479edda7a0612db288df81dfeb7'
```

Obtain the [MANO models](https://mano.is.tue.mpg.de/) separately and pass the directory
containing `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` as `task.mano_root`.

Return to the [quick start](../README.md#quick-start) after installation.
