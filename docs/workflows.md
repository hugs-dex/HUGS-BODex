# Workflow Guide

Start with [installation](installation.md) and the [bounded quick start](../README.md#quick-start).
This guide covers full runs and advanced options. All commands run from the BODex
root unless stated otherwise. Full runs below remove the quick start's scene limit.

For a first run, use `task.end=10` and `'task.gpus=[0]'`. Append
`task.dry_run=true` to inspect the plan without constructing the solver or saving
grasps. Scale and height filters can leave fewer scenes, including zero; inspect
the plan and increase the limit or choose an eligible scene set if needed.
A dry-run produces no grasps for the viewer or Bench.

## Prepare data

Download the object assets from the
[HUGS dataset](https://huggingface.co/datasets/MingruiYu/HUGS).
Under `object_assets/DGN_2k/`, the release provides:

- `DGN_2k_processed_scene_cfg_valid_split.tar.gz`: object meshes, collision assets,
  scene configurations, and splits required for synthesis.
- `DGN_2k_three_realsense_d435_random4096.tar.gz`: partial point clouds for workflows
  that use camera observations.

Follow the dataset card for extraction instructions, checksums, and usage terms.
Arrange the extracted collection under `$HUGS_DATASET_ROOT/object/DGN_2k` (a symlink
to the extracted collection is also supported). Point `HUGS_OBJECT_ROOT` at the
directory containing `scene_cfg/` and `processed_data/`:

```text
hugs-dataset/
└── object/
    └── DGN_2k/                 # HUGS_OBJECT_ROOT
        ├── scene_cfg/          # Per-scene NPY files
        ├── processed_data/     # Object meshes, collision assets, and point clouds
        ├── valid_split/        # Object splits
        └── vision_data/        # When the partial-point-cloud archive is extracted
            └── three_realsense_d435_random4096/
```

```bash
export HUGS_DATASET_ROOT=/path/to/hugs-dataset
export HUGS_OBJECT_ROOT="$HUGS_DATASET_ROOT/object/DGN_2k"
```

Before synthesis, generate the object-height table used by the default height
filter (skip this step if the table already exists):

```bash
python scripts/filter_tabletop_scene_cfg_by_height.py --datasets DGN_2k
```

This writes `tabletop_scene_object_heights.jsonl` under `HUGS_OBJECT_ROOT`, which
must be writable. The published archive does not include this generated table.

For a read-only dataset, write the table to a local directory instead:

```bash
mkdir -p "$PWD/outputs/scene_heights"
python scripts/filter_tabletop_scene_cfg_by_height.py --datasets DGN_2k \
  --output-stem "$PWD/outputs/scene_heights/DGN_2k"
```

Then append
`"+task.scene_source.object_height_record_path=$PWD/outputs/scene_heights/DGN_2k.jsonl"`
to synthesis commands. The leading `+` is required for this optional Hydra key.

Keep the data directory structure intact: scene files reference object assets by
relative path. Record the dataset revision used for each experiment. The synthesis
commands below pass the scene glob explicitly, so data can be stored outside the
repository. Results, data, and MANO models are not included in the source checkout.

Human initialization additionally requires an exported prior tree matching the scene
IDs. Set `task.human_prior.root` to that tree's suite-specific directory; see the
[prior format](contracts.md#human-prior-to-synthesis). Consuming an export does
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

Examples use `surface_demo` and `human_demo`. Use a new name for each experiment.

### Suite and baseline selection

Set `task.suite_config` and `task.baseline_config` together using a matching pair
below. The suite defines the grasp modes and their manipulation configs; the
baseline overrides settings within that suite. A baseline does not automatically
select a suite, and their `robot_name` values must match.

| Baseline | `task.baseline_config` with `task.suite_config=sim_shadow.yml` | `task.baseline_config` with `task.suite_config=sim_leap_sp.yml` |
| --- | --- | --- |
| Fixed contact mode | `heur_fix_shadow.yml` | `heur_fix_leap_sp.yml` |
| One contact mode per object scale | `heur_single_shadow.yml` | `heur_single_leap_sp.yml` |
| Multiple contact modes per object scale | `heur_multi_shadow.yml` | `heur_multi_leap_sp.yml` |
| Human prior | `human_shadow.yml` | `human_leap_sp.yml` |

The `heur_*` baselines require `task.init_source=surface_sample` (the default) and
`task.scene_source.use_object_scale_list=true`. The `human_*` baselines require
`task.init_source=human` and `task.scene_source.use_object_scale_list=false` (the
default), plus a matching `task.human_prior.root`. These requirements are validated,
not set automatically by the baseline.

Suite filenames are relative to `src/curobo/content/configs/manip/`; baseline
filenames are relative to `src/curobo/content/configs/baselines/`. Include `.yml`.
Use `task.baseline_config=null` to run without baseline overrides.

### Surface initialization

Generate grasps for all eligible scenes and all five grasp types with Shadow Hand:

```bash
python example_grasp/main.py task=synthesis \
  name=surface_demo \
  task.suite_config=sim_shadow.yml \
  task.baseline_config=heur_multi_shadow.yml \
  task.grasp_types=all \
  task.scene_source.use_object_scale_list=true \
  'task.gpus=[0,1]' \
  "task.scene_source.template_path=$HUGS_OBJECT_ROOT/scene_cfg/**/tabletop_ur10e/*.npy"
```

The command runs actual synthesis with surface initialization and no scene-count
limit. All other settings use their defaults. For a smaller validation run, append
`task.end=1000`: this randomly selects 1,000 scenes (using the default shuffle seed
123) before applying scale and height filters. The actual number used can be
smaller, including zero for a mode with no eligible scenes. For Leap-SP, change both
`task.suite_config=sim_leap_sp.yml` and
`task.baseline_config=heur_multi_leap_sp.yml`.

### Human-prior initialization

After [exporting priors with DexLearn](https://github.com/hugs-dex/HUGS-DexLearn#human-prior),
set the exported prior directory and run synthesis with Shadow Hand. The path below
assumes sibling checkouts and the default DexLearn output directory:

```bash
export HUGS_PRIOR_ROOT="$(realpath ../HUGS-DexLearn/output/humanMulti_humanMultiHierar_human_prior_full/obj_human_prior/step_007500_000100/DGN_2k/shadow_hand)"

python example_grasp/main.py task=synthesis \
  name=human_demo \
  task.suite_config=sim_shadow.yml \
  task.baseline_config=human_shadow.yml \
  task.grasp_types=all \
  task.scene_source.use_object_scale_list=false \
  task.init_source=human "task.human_prior.root=$HUGS_PRIOR_ROOT" \
  'task.gpus=[0,1]' \
  "task.scene_source.template_path=$HUGS_OBJECT_ROOT/scene_cfg/**/tabletop_ur10e/*.npy"
```

For a smaller validation run, append `task.end=1000`.

### Multiple GPUs

Add `'task.gpus=[0,1]'` to either synthesis command to distribute planned batches
across two GPUs. A single ID such as `'task.gpus=[0]'` selects one GPU.
`task.gpus=null` runs in the current process and respects `CUDA_VISIBLE_DEVICES`.
Explicit `task.gpus` IDs become worker `CUDA_VISIBLE_DEVICES` values; use physical
GPU IDs consistently. Each worker writes a separate log in the Hydra run directory,
and a failed worker causes the parent command to fail.

### Outputs and resuming

Results are written to the default directory under the repository:

```text
src/curobo/content/assets/output/<manipulation path without .yml>/<name>/graspdata/<scene_id>_grasp.npy
```

Use a new `name` for a separate experiment. `task.skip=true` resumes by skipping
existing NPY files; with `task.skip=false`, an existing experiment may prompt for
cleanup. Saving accepts `task.save_mode=npy` (default) or `none`.

The default saves solver stage poses and a derived squeeze pose (three poses in the
standard configs). Set `task.save_debug=true` to retain optimization trajectories,
and choose the saved subset with `task.save_data=final_and_mid` or `all`.
See [result fields and coordinate conventions](contracts.md#results-to-viewingevaluation).

## Evaluate with DexGraspBench

From a sibling HUGS-DexGraspBench checkout, pass the BODex output root and the
same run name to the [BODex workflow](https://github.com/hugs-dex/HUGS-DexGraspBench#bodex-grasps).
The root is `src/curobo/content/assets/output/` inside BODex by default, or
`HUGS_OUTPUT_ROOT` if set. It contains the suite/manipulation/run subdirectories;
do not pass an individual `graspdata/` directory to the batch wrapper.

## Visualization

The viewers use the default output directory shown above.

### Browse grasps

```bash
python example_grasp/main.py task=visualize name=surface_demo \
  task.suite_config=sim_shadow.yml task.grasp_types=all \
  task.port=8081
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

## Developer tools

Asset inspection tools live in `scripts/`: `vis_collision_spheres.py`,
`vis_self_collision_pairs.py`, and `vis_hand_pose_transfer.py`.
