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

Real human priors were also run for both suites in single and two-GPU modes (20 NPY
records total). Budget, type ID/name, sample indices, replacement masks, transfer
files, seed configurations and scene provenance passed the checks in
`validation/2026-09-24/check_human_results.py`. The historical prior producer commit
is unknown and is recorded as such; this is a functional smoke, not a versioned
research comparison.

The private fixed-input reference ran all ten surface-mode jobs successfully and its
NPY schema/finite-pose checks passed. This one-scene bounded comparison does not claim
quality, performance, or numerical equivalence. Failure logs, commands, hashes and
summaries are retained in `../validation/2026-09-24/` outside Git.

Release gates remain: `utils_python` has no license file and needs owner terms;
robot-asset redistribution terms need confirmation; the BODex object archive and a
clean anonymous public checkout need a successful fetch with recorded digest; and no
remote HUGS-BODex repository has been created or published. This candidate has not
been published.
