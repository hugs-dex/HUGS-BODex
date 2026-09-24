# HUGS-BODex

Grasp synthesis with surface sampling or exported human priors. The cuRobo package,
Hydra tasks, optimization algorithms and result layout are retained.

This is a **local release candidate**, not a validated public release. See
[validation status](docs/validation.md). The inherited [NVIDIA license](LICENSE),
section 3.3, restricts use to non-commercial research or evaluation. This is
source-available software, not OSI open source. Assets have separate terms; see
[notices](THIRD_PARTY_NOTICES.md).

## Installation and inputs

See [installation](docs/installation.md) and [contracts](docs/contracts.md).
The intended repository is `https://github.com/hugs-dex/HUGS-BODex`; this candidate
does not imply it has been published. Initialize the fixed HTTPS dependencies with
`git submodule update --init --recursive` after obtaining the checkout.

## Migration scope

Both suites retain all five mode IDs and both initialization paths. Runtime validation
is tracked separately from code migration. Paths below are relative to
`src/curobo/content/configs/manip`.

| ID | Mode | Shadow | Leap-SP |
| --- | --- | --- | --- |
| 1 | `1_right_two` | `sim_shadow/tabletop_two.yml` | `sim_leap_sp/tabletop_two.yml` |
| 2 | `2_right_three` | `sim_shadow/tabletop_three.yml` | `sim_leap_sp/tabletop_three.yml` |
| 3 | `3_right_full` | `sim_shadow/tabletop_full.yml` | `sim_leap_sp/tabletop_full.yml` |
| 4 | `4_both_three` | `sim_dual_dummy_arm_shadow/tabletop_three.yml` | `sim_dual_dummy_arm_leap_sp/tabletop_three.yml` |
| 5 | `5_both_full` | `sim_dual_dummy_arm_shadow/tabletop_full.yml` | `sim_dual_dummy_arm_leap_sp/tabletop_full.yml` |

Suites: `sim_shadow.yml`, `sim_leap_sp.yml`. Other inherited robots/configurations
are outside this formal support matrix.

## Synthesis

Run from the installed repository root:

```bash
python example_grasp/main.py task=synthesis name=surface_demo \
  task.suite_config=sim_shadow.yml task.grasp_types=all \
  task.init_source=surface_sample task.gpus=null \
  task.scene_source.template_path='/path/to/DGN_2k/scene_cfg/**/tabletop_ur10e/*.npy' \
  task.start=0 task.end=1 task.parallel_world=1 task.dry_run=true
```

Set `task.dry_run=false` to synthesize; use `sim_leap_sp.yml` for Leap-SP.
`task.grasp_types=all` covers all five entries; a single ID/name or a list such as
`'task.grasp_types=[1_right_two,5_both_full]'` selects a subset.
`task.scene_source.use_object_scale_list=true` applies per-type configured scales;
`false` disables that additional scale-list filter.

`task.gpus=null` runs in the current process and respects `CUDA_VISIBLE_DEVICES`.
`'task.gpus=[0]'` binds to the specified device; `'task.gpus=[0,1]'` launches the native
multi-GPU scheduler. Explicit IDs become worker `CUDA_VISIBLE_DEVICES` values, not
indices remapped through an outer visibility list. Use physical IDs consistently.
Planned batches are assigned once; grouping and solver reuse options are preserved.
Workers have separate logs in the Hydra run directory; any failed worker fails the run.

```bash
python example_grasp/main.py task=synthesis name=human_demo \
  task.suite_config=sim_shadow.yml task.grasp_types=all task.init_source=human \
  task.human_prior.root=/path/to/exported/prior \
  'task.gpus=[0,1]' task.start=0 task.end=2 task.skip=false
```

Human prior reading, budget allocation, sample indices, transfer, jitter and provenance
are included. Consuming an export requires no training checkout/checkpoint. Local
real-prior smoke results are recorded in the validation status; the producer commit
for those historical exports remains unknown.

Saving accepts `task.save_mode=npy` or `none`. `task.skip=true` skips existing NPY;
`none` never skips based on files. `task.save_debug=true` records optimization stages;
`task.save_data=final_and_mid` or `all` selects stages. Without debug saving results
contain the solver stage poses plus the derived squeeze pose; the standard configs
currently produce three poses. Removed initialization/export formats are rejected.

Results default to `src/curobo/content/assets/output/<manip path without .yml>/<name>/graspdata`.
Set `HUGS_OUTPUT_ROOT` to place them outside the checkout. Fix seed, config, input
hashes, environment and revision for comparisons.

### Compatibility launcher

```bash
bash scripts/run_all_grasps_multi_gpu.sh --hand leap_sp \
  --exp-name legacy_demo --parallel-env 1 --gpus 0 1 --start 0 --end 2
```

The shell stops at the first failure; `-k` disables skip. Legacy `--hand leap` retains
its original mapping outside the formal matrix. Legacy configs must supply a nonempty
`world.template_path`; this shell does not apply Hydra scene overrides. Ranges are
assigned before skip checks, with no redistribution. Use Hydra for the full
initialization matrix. The legacy human script retains `heuristic`/`human_prior`
aliases; Hydra accepts only `surface_sample`/`human`.

## Viewing and rendering

```bash
python example_grasp/main.py task=visualize name=surface_demo device=cpu \
  task.suite_config=sim_shadow.yml task.grasp_types=all task.host=127.0.0.1 task.port=8081
python example_grasp/main.py task=visualize_prior_path name=human_demo device=cpu \
  task.suite_config=sim_leap_sp.yml task.grasp_types=all task.mano_root=/path/to/mano/models \
  task.host=127.0.0.1 task.port=8082 task.show_text=true task.show_caption=true \
  'task.next_button_label=Next Batch' task.exclude_both_three=false
```

Forward the selected port when remote. The prior viewer defaults to a sparse figure
layout; the example enables navigation labels. MANO model files require separate
acquisition and licensing and are not included.

Offline render selects one manipulation path from the table, not a suite:

```bash
PYOPENGL_PLATFORM=egl python example_grasp/main.py task=render name=surface_demo \
  manip_cfg_file=sim_leap_sp/tabletop_full.yml device=cpu n_worker=1 \
  'task.sample_lst=[0]' task.b_opt_process=true 'task.opt_progress=[1.0]'
```

Repeat with any table path. `b_opt_process=true` chooses saved trajectory indices;
`[1.0]` also supports final-only results. The three-stage mode (`false`) requires three
saved poses and reports a diagnostic otherwise. Images are written beside `graspdata`
under `grasp_imgs`. Configure `EGL_DEVICE_ID` as needed; set `PYOPENGL_PLATFORM=egl`
before importing OpenGL. For moved results use `HUGS_PATH_MAP` in the contracts.
Asset maintenance tools remain in `scripts/`: `vis_collision_spheres.py`,
`vis_self_collision_pairs.py`, `vis_hand_pose_transfer.py`.

## Checks

```bash
python scripts/check_public_candidate.py
python -m pytest -q tests/test_public_contracts.py
```

Static checks validate source/YAML syntax and the ten-mode asset closure. They do not
establish synthesis quality or performance.
