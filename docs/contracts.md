# Input and output contracts

## Object assets to synthesis

Scene producers export NPY dictionaries and meshes/collision assets. Set
`task.scene_source.template_path` to an absolute glob or the default asset-relative
`object/DGN_2k/scene_cfg/**/tabletop_ur10e/*.npy`. A dataset mount/symlink may occupy
that location; datasets are excluded from Git/packages. Use the object assets from
[HUGS](https://huggingface.co/datasets/MingruiYu/HUGS), including
`object_assets/DGN_2k/DGN_2k_processed_scene_cfg_valid_split.tar.gz` for synthesis.
Archive selection and the local layout are described in the
[README](../README.md#prepare-data).

Scene fields retain `scene_id`, `scene`, `task`, object identity and paths. Object
`pose` is translation in metres followed by WXYZ quaternion in the scene frame;
`scale` is dimensionless mesh scale. Mesh/URDF paths resolve relative to the scene
file. Consumers do not recenter, mirror, normalize or renumber these inputs. Record
producer version and input hashes with every run.

## Human prior to synthesis

`HumanPriorSeedBuilder` consumes exported per-scene priors with `index_mcp_pos`,
`wrist_quat`, `active_hand_mask` and `budget_scores`. BODex allocates seed budgets
from those scores using its runtime budget settings. Set `task.human_prior.root`
to the export's `manifest.json` `scene_dir` (adjusting relocated paths), so
`<root>/<scene_id>.npy` exists. For DexLearn's `prior_full_20261007` export this is
`obj_human_prior/step_007500_000100/DGN_2k/shadow_hand` under the experiment output,
not either `_type/ckpts` or `_diffusion/ckpts`. Preserve scene IDs, type/sample axes,
type IDs 1–5 and the suite's hand indices: `[0]` for single right hand, `[0,1]` for dual hands. Wrist
quaternions use WXYZ and translations use metres in the scene frame. Transfer files,
budget rounding, replacement, scene/type seed derivation and jitter are inherited.

Record the prior producer commit, input hashes, generating configuration and
checkpoint. Consuming the export does not require the producer's training checkout.
The `prior_full_20261007` export's manifest records the producer commit and both
checkpoint hashes, with pose step `007500`, score step `000100`,
`robot_name=shadow_hand` and `robot_size=1.0`. Check the manifest's coordinate
contract and hand-size ratio when choosing a prior for another suite; BODex's
record validator checks the arrays and scene ID, but does not enforce the export's
robot namespace or hand-size ratio.

## Results to viewing/evaluation

NPY dictionaries retain `robot_pose`, `joint_names`, `world_cfg`, `scene_path`,
`manip_name`, contact/error fields, optional `debug_info`, `init_source` and human
provenance. DoF order follows `joint_names`; root poses where used are translation
plus WXYZ quaternion followed by joints. Use the robot config and saved joint names,
not array length, to interpret coordinates. Human provenance retains scene file,
IDs/names/budgets, selected sample indices, replacement mask, transfer file and
`init_seed_config`. Only projection-specific metadata and USD format are removed.

`save_debug=false` saves solver stage poses and the derived squeeze pose (three
poses in the standard configs). Debug saving and `save_data` select optimization
trajectory subsets; a final-only subset is distinct from this default.
`none` emits no grasp NPY and cannot satisfy skip. Viewers display only saved stages.
Synthesis and viewers use `src/curobo/content/assets/output/` by default.

With `HUGS_DATASET_ROOT` set, saved dataset paths are relative to that root and
records include `path_root: "HUGS_DATASET_ROOT"`. Set the variable to the new bundle
location when viewing relocated results. `HUGS_OBJECT_ROOT` selects an object
collection for synthesis; it is not the root for saved references.

Relocate existing absolute viewer/render inputs using explicit path-component mappings:

```bash
export HUGS_PATH_MAP='{"/old/assets":"/new/assets","/old/prior":"/new/prior"}'
```

The longest prefix wins, even if the old path exists. Metadata and coordinate/scale
semantics are unchanged. Relative object meshes follow the relocated scene directory.
There is no implicit match against a private repository name.
