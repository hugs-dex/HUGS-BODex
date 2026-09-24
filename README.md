# HUGS-BODex

GPU-accelerated dexterous grasp synthesis for **HUGS**. HUGS-BODex optimizes
single-hand and bimanual grasps for Shadow Hand and Leap-SP, using either surface
samples or exported human priors. It supports multiple GPUs, interactive result
viewing, and offline rendering.

[Project website](https://hugs-dex.github.io/) · [Input/output reference](docs/contracts.md)

## Installation

### Requirements

- Linux x86-64 with Conda, Git, and a C++ compiler with OpenMP support (GCC/G++ 11 recommended).
- An NVIDIA GPU and a driver compatible with CUDA 12.1.
- A CUDA 12.x toolkit with `nvcc` on `PATH`. The toolkit is needed to compile
  extensions; installing PyTorch alone does not provide it.

This installation was tested on Ubuntu 22.04 with GCC 11.4, CUDA toolkit 12.4,
and an RTX 4090.

The installation below uses Python 3.10, PyTorch 2.2.2 with CUDA 12.1, NumPy 1.26.4,
Coal 3.0.1, and Eigen 3.4. Run all commands from the repository root unless noted.

### Environment and dependencies

```bash
git clone --recurse-submodules https://github.com/hugs-dex/HUGS-BODex.git
cd HUGS-BODex

conda create -n hugs-bodex --override-channels -c conda-forge \
  python=3.10 numpy=1.26.4 coal=3.0.1 eigen=3.4 pip -y
conda activate hugs-bodex

python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install 'setuptools<81' wheel setuptools-scm pybind11 ninja
python -m pip install torch-scatter==2.1.2+pt22cu121 \
  --no-deps --find-links https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

For an existing checkout, initialize its pinned dependencies with
`git submodule update --init --recursive`. Both `pytorch_kinematics` and
`utils_python` are required.

### Build and install

```bash
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS=4

python -m pip install --no-build-isolation \
  -e third_party/pytorch_kinematics -e third_party/utils_python -e '.[render]'
(cd src/curobo/geom/cpp && python -m pip install . --no-build-isolation)
```

Keep the Conda environment activated when building the Coal extension: its headers
and libraries are found through `CONDA_PREFIX`. By default, the CUDA build targets
visible GPUs. For a machine without a visible GPU, set `TORCH_CUDA_ARCH_LIST` to the
target architecture before building, for example `8.9` for an RTX 4090. Reduce
`MAX_JOBS` if compilation runs out of memory.

### Optional: human-prior visualization

Human-initialized synthesis only needs exported prior files. To also display the
human hand meshes in the prior viewer, install:

```bash
python -m pip install --no-build-isolation \
  'manopth @ git+https://github.com/DexGrasp-TH/manopth.git@dd83a157dccd4479edda7a0612db288df81dfeb7' \
  chumpy==0.70 opencv-python-headless==4.10.0.84
```

Obtain the [MANO models](https://mano.is.tue.mpg.de/) separately and pass the directory
containing `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` as `task.mano_root`.

## Prepare data

Download `DGN_2k_processed.zip` from the
[BODex object dataset](https://huggingface.co/datasets/JiayiChenPKU/BODex) and extract
it. Point `HUGS_DATA_ROOT` at the directory containing `scene_cfg/` and
`processed_data/`:

```text
DGN_2k/
├── scene_cfg/          # Per-scene NPY files
└── processed_data/     # Object meshes, collision assets, and point clouds
```

```bash
export HUGS_DATA_ROOT=/path/to/DGN_2k
export HUGS_OUTPUT_ROOT="$PWD/outputs"
```

Keep the data directory structure intact: scene files reference object assets by
relative path. The synthesis commands below pass the scene glob explicitly, so data
can be stored outside the repository. Results, data, and MANO models are not included
in the source checkout.

Human initialization additionally requires an exported prior tree matching the scene
IDs. Set `task.human_prior.root` to that tree's suite-specific directory; see the
[prior format](docs/contracts.md#human-prior-to-synthesis). Consuming an export does
not require the training repository or a training checkpoint.

## Supported hands and grasp modes

Select `sim_shadow.yml` for Shadow Hand or `sim_leap_sp.yml` for Leap-SP. Both suites
provide the same five mode IDs:

| ID | Mode | Shadow manipulation | Leap-SP manipulation |
| --- | --- | --- | --- |
| 1 | `1_right_two` | `sim_shadow/tabletop_two.yml` | `sim_leap_sp/tabletop_two.yml` |
| 2 | `2_right_three` | `sim_shadow/tabletop_three.yml` | `sim_leap_sp/tabletop_three.yml` |
| 3 | `3_right_full` | `sim_shadow/tabletop_full.yml` | `sim_leap_sp/tabletop_full.yml` |
| 4 | `4_both_three` | `sim_dual_dummy_arm_shadow/tabletop_three.yml` | `sim_dual_dummy_arm_leap_sp/tabletop_three.yml` |
| 5 | `5_both_full` | `sim_dual_dummy_arm_shadow/tabletop_full.yml` | `sim_dual_dummy_arm_leap_sp/tabletop_full.yml` |

Manipulation paths are relative to `src/curobo/content/configs/manip/`.
`task.grasp_types=all` selects all five modes. A single ID/name or a list such as
`'task.grasp_types=[1_right_two,5_both_full]'` selects a subset.

## Grasp synthesis

### Surface initialization

Start with one scene and inspect the plan:

```bash
python example_grasp/main.py task=synthesis name=surface_demo \
  task.suite_config=sim_shadow.yml task.grasp_types=all \
  task.init_source=surface_sample task.gpus=null \
  "task.scene_source.template_path=$HUGS_DATA_ROOT/scene_cfg/**/tabletop_ur10e/*.npy" \
  task.start=0 task.end=1 task.parallel_world=1 task.dry_run=true
```

Replace `task.dry_run=true` with `task.dry_run=false` to generate grasps. Change
`task.suite_config` to `sim_leap_sp.yml` for Leap-SP. `task.start` is inclusive and
`task.end` is exclusive; scene selection uses `task.scene_source.shuffle_seed=123`
by default. Set `task.scene_source.shuffle_before_slice=false` for sorted selection.

### Human-prior initialization

```bash
python example_grasp/main.py task=synthesis name=human_demo \
  task.suite_config=sim_shadow.yml task.grasp_types=all \
  task.init_source=human task.human_prior.root=/path/to/exported/prior/shadow \
  "task.scene_source.template_path=$HUGS_DATA_ROOT/scene_cfg/**/tabletop_ur10e/*.npy" \
  task.gpus=null task.start=0 task.end=1 task.parallel_world=1
```

Budgets are allocated across the five types using the prior's scores.
`task.human_prior.total_budget`, `min_type_budget`, `max_type_budget`, and
`budget_resolution` control allocation. Outputs retain selected prior sample indices,
transfer settings, and initialization poses for inspection.

### Multiple GPUs

Add `'task.gpus=[0,1]'` to either synthesis command to distribute planned batches
across two GPUs. A single ID such as `'task.gpus=[0]'` selects one GPU.
`task.gpus=null` runs in the current process and respects `CUDA_VISIBLE_DEVICES`.
Explicit `task.gpus` IDs become worker `CUDA_VISIBLE_DEVICES` values; use physical
GPU IDs consistently. Each worker writes a separate log in the Hydra run directory,
and a failed worker causes the parent command to fail.

### Outputs and resuming

Results are written to:

```text
$HUGS_OUTPUT_ROOT/<manipulation path without .yml>/<name>/graspdata/<scene_id>_grasp.npy
```

When `HUGS_OUTPUT_ROOT` is unset, the root is `src/curobo/content/assets/output`.
Use a new `name` for a separate experiment. `task.skip=true` resumes by skipping
existing NPY files; with `task.skip=false`, an existing experiment may prompt for
cleanup. Saving accepts `task.save_mode=npy` (default) or `none`.

The default saves solver stage poses and a derived squeeze pose (three poses in the
standard configs). Set `task.save_debug=true` to retain optimization trajectories,
and choose the saved subset with `task.save_data=final_and_mid` or `all`.
See [result fields and coordinate conventions](docs/contracts.md#results-to-viewingevaluation).

## Visualization

Keep `HUGS_OUTPUT_ROOT` set to the same directory used for synthesis.

### Browse grasps

```bash
python example_grasp/main.py task=visualize name=surface_demo device=cpu \
  task.suite_config=sim_shadow.yml task.grasp_types=all \
  task.host=127.0.0.1 task.port=8081
```

Open `http://127.0.0.1:8081` in a browser. When running remotely, forward the chosen
port over SSH. Match `name` and `task.suite_config` to the synthesis run.

### Inspect human prior → initialization → result

```bash
python example_grasp/main.py task=visualize_prior_path name=human_demo device=cpu \
  task.suite_config=sim_shadow.yml task.grasp_types=all \
  task.mano_root=/path/to/mano/models task.host=127.0.0.1 task.port=8082 \
  task.show_text=true task.show_caption=true 'task.next_button_label=Next Batch' \
  task.exclude_both_three=false
```

The viewer uses saved provenance to locate the prior, point cloud, and scene. If an
export contains paths from another machine, supply explicit prefix mappings:

```bash
export HUGS_PATH_MAP='{"assets/object/DGN_2k":"/path/to/DGN_2k","/old/prior":"/path/to/exported/prior"}'
```

### Render images

Rendering uses pyrender and requires EGL/OpenGL libraries and a working graphics
driver. Select one manipulation configuration from the table above:

```bash
PYOPENGL_PLATFORM=egl python example_grasp/main.py task=render name=surface_demo \
  manip_cfg_file=sim_shadow/tabletop_full.yml device=cpu n_worker=1 \
  'task.sample_lst=[0]' task.b_opt_process=true 'task.opt_progress=[1.0]'
```

JPGs are written under `grasp_imgs/` beside `graspdata/`. Set `EGL_DEVICE_ID` to choose
an EGL device when needed. `[1.0]` renders the last saved pose and supports final-only
results. The alternate three-stage mode (`task.b_opt_process=false`) requires three
saved poses.

## Compatibility and developer tools

The Hydra synthesis task is the recommended entry point for both initialization
methods. The original launcher is also available:

```bash
bash scripts/run_all_grasps_multi_gpu.sh --hand leap_sp \
  --exp-name legacy_demo --parallel-env 1 --gpus 0 1 --start 0 --end 2
```

Before using it, set `world.template_path` in the selected manipulation configs.
This launcher uses those paths, not Hydra scene overrides. It assigns ranges before
skip checks; `-k` disables skip. Use Hydra for the full surface/human support matrix.

Asset inspection tools live in `scripts/`: `vis_collision_spheres.py`,
`vis_self_collision_pairs.py`, and `vis_hand_pose_transfer.py`.

## Acknowledgements and license

HUGS-BODex builds on [BODex](https://github.com/JYChen18/BODex),
[cuRobo](https://github.com/NVlabs/curobo),
[pytorch_kinematics](https://github.com/DexGrasp-TH/pytorch_kinematics), and
[utils_python](https://github.com/Mingrui-Yu/utils_python).
The inherited [NVIDIA license](LICENSE) restricts use to non-commercial research or
evaluation. See [asset notices](LICENSE_ASSETS) and
[third-party acknowledgements](THIRD_PARTY_NOTICES.md) for the included components.
