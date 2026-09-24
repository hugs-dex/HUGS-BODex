# Candidate validation

Migration includes Shadow/Leap-SP, IDs 1–5, surface/human initialization, native and
legacy multi-GPU execution, result/prior viewers and offline render.

Completed: source/YAML syntax, ten-mode config/URDF/mesh/transfer/sphere closure,
anonymous access and fetch of both fixed Git dependencies. Robot asset closure
contains no LFS pointers. See `scripts/check_public_candidate.py`.

The isolated environment built the CUDA extension and Coal wrapper; `pxr` is absent.
CPU contract tests pass (`8 passed`), and the controlled worker-failure test confirms
non-zero worker propagation. Surface sampling produced 20 NPY records across the two
suites and five modes in single and real two-GPU runs; the result checker validated
400 finite candidate poses and loaded robot/object meshes. Offline EGL rendering
produced four JPGs. The ordinary viewer returned HTTP 200 for both suites. The prior
viewer loaded all ten suite/mode combinations with a real MANO export and returned
HTTP 200; the input point-cloud path required the documented `HUGS_PATH_MAP` mapping.
Synchronous geometry checks generated finite MANO, object, initial/final robot meshes
and point clouds for all ten modes. These are automated checks; interactive browser
visual inspection and the standalone asset-maintenance UIs have not been completed.

Real human priors were also run for both suites in single and two-GPU modes (20 NPY
records total). Budget, type ID/name, sample indices, replacement masks, transfer
files, seed configurations and scene provenance passed the checks in
`../validation/2026-09-24/check_human_results.py`. The historical prior producer commit
is unknown and is recorded as such; this is a functional smoke, not a versioned
research comparison.

The private fixed-input reference ran all ten surface jobs and all ten human jobs.
Twenty candidate/reference pairs match in schema, joint order, world pose/scale and
array shapes. Human budgets, sample indices and transfer file contents also match.
The dual-full human initial poses differ by up to 0.000246 in the saved coordinate
vector; final optimizer poses/errors also differ. No numerical acceptance threshold
was established, so numerical equivalence remains unverified. Error summaries are
recorded in `reference_contract_comparison.json`.

The reference uses the fixed private Python source with the same dependency/extension
environment and a separately targeted USD install. An explicit configuration overlay
sets the four previously null dual-hand `sampling_mode` values to
`random_xy_symmetric`, matching the candidate fix. This is not a clean-host install
comparison or an unmodified-default performance benchmark. The candidate environment
was made from an isolated Conda clone and its extensions rebuilt. Failure logs, commands, hashes and
summaries are retained in `../validation/2026-09-24/` outside Git.

Release gates remain: `utils_python` has no license file and needs owner terms;
robot-asset redistribution terms need confirmation; the BODex object archive and a
clean anonymous public checkout need a successful fetch with recorded digest; and no
remote HUGS-BODex repository has been created or published. This candidate has not
been published.
