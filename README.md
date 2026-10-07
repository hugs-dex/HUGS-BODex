<h1 align="center">HUGS-BODex</h1>

<p align="center">Generate dexterous robot grasps with surface samples or learned human priors.</p>

<p align="center">
  <a href="https://github.com/hugs-dex/HUGS-Main">HUGS Project</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#full-dataset-synthesis">Full Dataset Synthesis</a> ·
  <a href="#documentation">Documentation</a>
</p>

GPU-accelerated grasp optimization for **Shadow Hand and Leap-SP**, supporting
single-hand and bimanual contact modes. Exported human priors from
[HUGS-DexLearn](https://github.com/hugs-dex/HUGS-DexLearn#human-prior) guide the
initial poses and mode budgets; surface initialization can run independently.

## Installation

Use **Linux x86_64, Python 3.10, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU,
and a CUDA 12.x toolkit**. Extension builds also require Git and a C++ compiler
with OpenMP support (GCC 11 recommended).

Run from the repository root:

```bash
git submodule update --init --recursive
uv venv --python 3.10 .venv
source .venv/bin/activate
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS=4
uv pip install pip setuptools wheel
uv pip install --no-deps --no-build-isolation chumpy==0.70
uv sync --frozen --extra render --extra human
uv pip install --no-deps \
  coal==3.0.1 cmeel==0.61.0 cmeel-assimp==5.4.3.1 \
  cmeel-boost==1.87.0.1 cmeel-octomap==1.10.0 cmeel-qhull==8.0.2.1 \
  cmeel-zlib==1.3.2 eigenpy==3.10.3
(cd src/curobo/geom/cpp && uv pip install . --no-build-isolation)
```

Keep `.venv` active for builds and subsequent commands. The separate Coal install
and wrapper build are required. Install chumpy separately because its build needs pip.
See the [installation guide](docs/installation.md)
for the tested stack, dependency pins, environment updates, and optional MANO viewer.

## Prepare Data

Download the [HUGS object assets](https://huggingface.co/datasets/MingruiYu/HUGS).
For synthesis, extract `DGN_2k_processed_scene_cfg_valid_split.tar.gz` under
`$HUGS_DATASET_ROOT/object/DGN_2k`, preserving `scene_cfg/` and `processed_data/`.

```bash
export HUGS_DATASET_ROOT=/path/to/hugs-dataset
export HUGS_OBJECT_ROOT="$HUGS_DATASET_ROOT/object/DGN_2k"

# Generate the height table once if it is absent from the extracted collection.
python scripts/filter_tabletop_scene_cfg_by_height.py --datasets DGN_2k
```

The height table is required by the default suites and is not included in the
archive. The command writes into the object collection; see [data preparation](docs/workflows.md#prepare-data)
for extraction details and the local-cache option for read-only datasets.

## Quick Start

Generate a small Shadow run from surface initialization, using one GPU and up to
10 scenes before scale and height filtering:

```bash
python example_grasp/main.py task=synthesis name=surface_demo \
  task.suite_config=sim_shadow.yml task.baseline_config=heur_multi_shadow.yml \
  task.init_source=surface_sample task.scene_source.use_object_scale_list=true \
  task.grasp_types=all task.end=10 'task.gpus=[0]' \
  "task.scene_source.template_path=$HUGS_OBJECT_ROOT/scene_cfg/**/tabletop_ur10e/*.npy"

python example_grasp/main.py task=visualize name=surface_demo \
  task.suite_config=sim_shadow.yml task.grasp_types=all task.port=8081
```

Open `http://127.0.0.1:8081`. Filtered scenes and saved record counts may be smaller;
if the selection is empty, increase the limit or select eligible scenes.
Append `task.dry_run=true` to the synthesis command to inspect the plan without
saving grasps. See [Full Dataset Synthesis](#full-dataset-synthesis) for full runs.

For Leap-SP, change the suite and baseline together to `sim_leap_sp.yml` and
`heur_multi_leap_sp.yml`. See [baseline and GPU options](docs/workflows.md#grasp-synthesis).

## Human Prior

First [train and export a Human Prior](https://github.com/hugs-dex/HUGS-DexLearn#human-prior),
or prepare a compatible export. With sibling checkouts and the DexLearn example's
default output directory:

```bash
export HUGS_PRIOR_ROOT="$(realpath ../HUGS-DexLearn/output/humanMulti_humanMultiHierar_human_prior_full/obj_human_prior/step_007500_000100/DGN_2k/shadow_hand)"

python example_grasp/main.py task=synthesis name=human_demo \
  task.suite_config=sim_shadow.yml task.baseline_config=human_shadow.yml \
  task.init_source=human task.scene_source.use_object_scale_list=false \
  "task.human_prior.root=$HUGS_PRIOR_ROOT" \
  task.grasp_types=all task.end=10 'task.gpus=[0]' \
  "task.scene_source.template_path=$HUGS_OBJECT_ROOT/scene_cfg/**/tabletop_ur10e/*.npy"

python example_grasp/main.py task=visualize name=human_demo \
  task.suite_config=sim_shadow.yml task.grasp_types=all task.port=8081
```

The export must match the scene IDs and target hand size. Consuming priors requires
neither the training checkpoint nor MANO; MANO is needed by the optional human-hand
viewer. Open `http://127.0.0.1:8081` to browse the synthesized grasps.

## Full Dataset Synthesis

After preparing the data above, use the complete launch commands in the workflow
guide for [surface initialization](docs/workflows.md#surface-initialization) or
[human-prior initialization](docs/workflows.md#human-prior-initialization).
These runs use `surface_full` and `human_full`, all five grasp modes, and no scene
slice limit. GPU selection, Leap-SP substitutions, and resuming are documented
alongside the commands.

Full runs process all eligible scenes matching the specified glob, subject to
scale and height filters and, for human initialization, prior budgets. Synthesis
does not automatically select a `valid_split`; human priors must cover the
eligible scene IDs.

## Outputs and Evaluation

By default, grasp records are saved under:

```text
src/curobo/content/assets/output/<manipulation path>/<name>/graspdata/
```

For example, the right-full run uses
`sim_shadow/tabletop_full/surface_demo/graspdata/` below that root.
`HUGS_OUTPUT_ROOT` overrides the root; `task.skip=true` skips existing records.
Use a new experiment name to keep separate runs.

Continue with [HUGS-DexGraspBench](https://github.com/hugs-dex/HUGS-DexGraspBench#bodex-grasps)
to convert, evaluate, and collect successful grasps. Pass the output root and the
same run name. Viewing a grasp does not establish simulation success.

## Documentation

- [Installation and troubleshooting](docs/installation.md)
- [Data preparation, synthesis, viewers, and rendering](docs/workflows.md)
- [Input/output contracts](docs/contracts.md)
- [Validation and known limitations](docs/validation.md)

## Acknowledgements and License

Built on [BODex](https://github.com/JYChen18/BODex) and
[cuRobo](https://github.com/NVlabs/curobo). The inherited [NVIDIA license](LICENSE)
restricts use to non-commercial research or evaluation. See [asset notices](LICENSE_ASSETS),
[third-party acknowledgements](THIRD_PARTY_NOTICES.md), and the [HUGS citation](https://github.com/hugs-dex/HUGS-Main#citation).
