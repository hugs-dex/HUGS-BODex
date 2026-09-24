# Input and output contracts

## Object assets to synthesis

Scene producers export NPY dictionaries and meshes/collision assets. Set
`task.scene_source.template_path` to an absolute glob or the default asset-relative
`object/DGN_2k/scene_cfg/**/tabletop_ur10e/*.npy`. A dataset mount/symlink may occupy
that location; datasets are excluded from Git/packages. The upstream asset URL is
https://huggingface.co/datasets/JiayiChenPKU/BODex (`DGN_2k_processed.zip`). Archive
version, digest, anonymous access and license remain required acquisition gates.

Scene fields retain `scene_id`, `scene`, `task`, object identity and paths. Object
`pose` is translation in metres followed by WXYZ quaternion in the scene frame;
`scale` is dimensionless mesh scale. Mesh/URDF paths resolve relative to the scene
file. Consumers do not recenter, mirror, normalize or renumber these inputs. Record
producer version and input hashes with every run.

## Human prior to synthesis

`HumanPriorSeedBuilder` consumes exported per-scene priors with `index_mcp_pos`,
`wrist_quat`, `active_hand_mask` and existing scores/budgets. Set
`task.human_prior.root`. Preserve scene IDs, type/sample axes, type IDs 1–5 and the
suite's hand indices: `[0]` for single right hand, `[0,1]` for dual hands. Wrist
quaternions use WXYZ and translations use metres in the scene frame. Transfer files,
budget rounding, replacement, scene/type seed derivation and jitter are inherited.

Record the prior producer commit, input hashes, generating configuration and
checkpoint. Consuming the export does not require the producer's training checkout.
Local validation used real exported priors for both suites and all five modes. The
historical export's producer commit is unknown, so it cannot establish a versioned
research comparison. Synthetic data cannot substitute for that provenance gate.

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
Set `HUGS_OUTPUT_ROOT` consistently for synthesis and viewing.

Relocate existing viewer/render input files using explicit path-component mappings:

```bash
export HUGS_PATH_MAP='{"/old/assets":"/new/assets","/old/prior":"/new/prior"}'
```

The longest prefix wins, even if the old path exists. Metadata and coordinate/scale
semantics are unchanged. Relative object meshes follow the relocated scene directory.
There is no implicit match against a private repository name.
