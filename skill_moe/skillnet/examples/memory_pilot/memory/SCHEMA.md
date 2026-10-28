# Executed motion value extraction

`MemoryStore.from_recording` supports three value sources:

- `feature` (default): use one vector row of an `(N,D)` NPZ feature matrix, usually a `feat__*` key. The
  key and value feature may be identical.
- `pred_chunk_executed`: take only `pred_chunk[row, :n_executed]`. An execution count
  larger than the recorded chunk horizon is an error. No predicted tail is consumed.
- `env_trajectory`: recover the observed motion for the executed steps from the `step`
  vector and episode-level `env_eef_pos`, `env_eef_quat`, and `env_gripper` arrays.

## Fixed-length representation

Executed segment lengths can vary, for example when an episode terminates partway through
a replan window. Sequence sources are linearly resampled along time to a fixed `value_length`.
The caller may set it explicitly; otherwise the manifest's `replan_steps` is used, with the
maximum `n_executed` across the entire run as fallback. Original segments are resampled
once, to the same length across episodes. Resampling aligns the first and last executed
samples with the first and last output samples; a one-step segment is repeated. The original
`n_executed` and `[start_step, end_step)` remain in metadata. Padding and masks are not used.
These are phase-normalized motion representations, not exact action playback: duration
must be read from metadata, and downsampling can discard intermediate details. With an
output length of one, the first executed sample is retained.

For `env_trajectory`, each environment sample becomes `[delta_position(3), relative_rotation
vector(3), delta_gripper(Dg)]`. Position and gripper are changes from the segment's starting
observation. Rotation is the shortest start-relative axis-angle vector, derived from xyzw
quaternions. Position deltas stay in the recorded world coordinate frame.
Position and gripper are interpolated linearly; orientation uses shortest-arc quaternion
SLERP before conversion to rotation vectors. This avoids inventing a zero rotation when
the observed orientation crosses the ±pi rotation-vector branch.

`step` denotes the policy observation's environment step. Because the recorder stores
post-action observations, the row at environment step `t` corresponds to `t -
env_step_offset`; the executed observations following that step are the next
`n_executed` rows. An explicit nonnegative `env_step_offset` may be supplied on the episode
or manifest. For format version 1, it is inferred as `episode.steps + 1 - len(env_eef_pos)`
and must be 0 or 1, matching G2's loader. If the origin cannot be determined or the arrays
do not contain both the segment start and all executed observations, extraction fails.

`MemoryMetadata.value_source` records the selected extraction mode and `value_length` records
the fixed sequence length (or `None` for a feature vector). The existing representation
version should also encode any caller-specific feature/model details.

## Exclusion and persistence contract

Initial-state IDs use the same JSON string tuple as G2's `load_run`:
`["suite", "task_id-or-task", "init_state_id"]`. Always construct them with
`namespace_init_state_id`, preserving numeric task ID zero. Metadata and search/exclusion
IDs are validated and canonicalized; bare local IDs are rejected. Format-v2 memory NPZs
use this convention. Format-v1 stores used incompatible IDs and must be rebuilt from
recordings; loading them is rejected rather than risking silent leakage.

Except in `none` mode, default exclusions require both query episode and initial-state
IDs. Entire matching episodes/states are removed, including adjacent and overlapping
windows. Explicit held-out exclusion sets and representation filters are applied before
fitting key statistics, identically for every retrieval mode. Store only the memory-side
split, or exclude **all** held-out episodes/states, not just the current query. The library
cannot infer which other stored entries belong to an evaluation split.

Duplicate episode IDs/files are rejected during recording import. Executed intervals may
not extend into the next inference or beyond recorded episode `steps`. A feature vector's
executed provenance is the caller's responsibility; use the executed extraction modes for
raw predictions/trajectories. Arbitrary flattened features cannot be checked for hidden
unexecuted prediction content by this library.
