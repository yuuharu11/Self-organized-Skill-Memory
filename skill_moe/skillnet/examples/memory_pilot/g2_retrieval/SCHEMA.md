# G2 retrieval schema and metric choices

The reader expects `<run_dir>/manifest.json` and one `<run_dir>/episodes/<episode_id>.npz`
for each entry in manifest `episodes`. `episodes` may be a list of objects containing
`episode_id` and `init_state_id`, or an object keyed by episode ID whose values contain
`init_state_id`. An entry may set `file`/`path` to an alternate relative NPZ path. Task,
suite, and success metadata are accepted but not needed by the aggregate retrieval score.

Each NPZ must contain `step` and `proprio` with one row per policy inference, one or more
`feat__<name>` matrices with the same number of rows, and environment-per-step vectors
`env_eef_pos (T,3)`, `env_eef_quat (T,4)`, and `env_gripper (T[,Dg])`. The row's `step`
is the environment step number at which its policy input was observed. Quaternion order is
xyzw. Environment arrays cover every environment step. Missing feature
rows are not supported; every requested feature must be present in every episode.
Incomplete feature coverage, nonfinite inputs, zero quaternions, inconsistent feature
dimensions, noninteger/unordered inference steps and duplicate episode files are errors.

By default, environment array index zero is observation step zero (before the first action).
An optional integer `env_step_offset` in an episode entry or the manifest declares the step
represented by environment row zero; an episode value takes precedence. For the current
`format_version=1` recorder, when no offset is supplied, episode `steps` is the number of
executed environment actions: `len(env_eef_pos)==steps` means post-action-only observations
(offset 1), while `len(env_eef_pos)==steps+1` includes reset (offset 0). Ambiguous v1 input
is rejected. The inference's environment index is `step-env_step_offset`; offsets actually
used are recorded in `results.json`. This avoids shifting the motion target by one action.

## Split and leakage

The sorted unique initial-state groups are shuffled using the configured seed. If suite/task
metadata is present, the group key is a JSON tuple of suite, task_id-or-task and init_state_id, because many
recorders number initial states locally within each task. Otherwise the raw `init_state_id`
is grouped with empty suite/task components. A rounded
`query_fraction` of whole IDs (at least one, at most all but one) is assigned to queries;
all their episodes follow. All other IDs and episodes form the retrieval store. The code
asserts both ID sets are disjoint and additionally excludes same-init-state and same-episode
candidates during scoring. Thus episodes sharing an initial-state ID cannot cross the split.
The `neighbor_exclusion_steps` configuration field is retained for recording compatibility;
same-episode candidates are excluded entirely, which is stricter than any adjacent-window
exclusion radius.

Rows without a complete future horizon are dropped (the final usable environment array
index is `T-H-1`). For
`H`, outcomes use offsets 1..H relative to the observed environment pose/gripper: position
displacement is `p[t+h]-p[t]`, orientation change is the xyzw quaternion
`conjugate(q[t]) * q[t+h]` in the starting end-effector frame, and gripper is
`g[t+h]-g[t]`. Query-to-retrieved outcome error is averaged over horizon: Euclidean norm
of displacement-vector difference, SO(3) geodesic angle between relative rotations (quaternion
sign is ignored), and mean absolute gripper-delta difference. They are reported separately
(meters, radians, and gripper units).

Feature and proprio kNN use identical query/store rows and Euclidean distance after
per-dimension standardization from the store rows only. At the kth-distance boundary,
scores integrate uniform selection among all tied neighbors, avoiding manifest-order bias.
Random retrieval reports the exact expected mean error of uniform sampling without
replacement; there is no single-draw noise. The motion oracle ranks separately per metric,
so it is an optimistic, component-wise upper bound rather than a realizable single ranking.
For k=5 with fewer than five eligible store rows, all available rows are used. `results.json`
records split membership, effective k, configuration and feature metadata. The code is
NumPy-only and needs no GPU.

## Uncertainty and the G2 evidence decision

For each feature/k/component, improvement is **proprio error minus feature error**, paired
on the same query. Bootstrap units are whole initial-state groups, including all repeat
episodes and overlapping windows within a group. The row-weighted mean is recomputed in
each resample; the number of windows is not treated as an independent sample size.
Intervals condition on the fixed retrieval store and split. They do not cover uncertainty
from changing the store/split, task distribution or choosing additional seeds after seeing
results. Representative task coverage and independent confirmation remain necessary for
deployment; checkpoint and feature provenance must be verified by the recorder.

With fewer than **10 usable query initial-state groups**, intervals are unavailable and
the evidence decision is `inconclusive`. A nonzero effect with no between-group variation
also yields no interval, rather than presenting a degenerate point mass as certainty.
Ten groups is a conservative minimum, not a guarantee of bootstrap coverage.

Otherwise, paired cluster percentile intervals use 95% familywise confidence with
Bonferroni adjustment across **all features × k values × three components** in this run.
The seeded bootstrap uses at least `bootstrap_samples` (CLI `--bootstrap-samples`, default
10000), increased automatically so that each adjusted tail has at least ten expected
resamples. Coverage is approximate, especially with few groups.

The conservative `clear_improvement` rule requires one feature to have, at every tested k,
a strictly positive interval lower bound in at least one component and nonnegative lower
bounds in all three components. Otherwise the decision is `inconclusive`, which must not
be interpreted as proof that features have no value. The raw component scores and paired
improvements remain available for review. Feature identity must be action-expert suffix
hidden state, with layer, diffusion timestep and pooling recorded in manifest metadata;
statistical superiority alone cannot verify this provenance.

## Cross-task and cross-subtask retrieval (`crosstask.py`)

`python -m skill_moe.skillnet.examples.memory_pilot.g2_retrieval.crosstask <run_dir>
[--plan-annotations data_process/libero/instruct2plan_40.json] [--output-dir DIR]`
(default output `memory-pilot/g2-crosstask-<UTC>`). It additionally needs manifest `task`
per episode and NPZ `skills`, `skill_mask`, `pred_chunk`, `n_executed`. Split, row
construction, motion target, metrics, store-only standardization, tie handling and the
cluster bootstrap are those above.

**Current subtask.** The policy receives the whole plan in every inference: the evaluation
client sends the task's `all_classes` from the annotation file and `LiberoInputs` stores
`ID+1` (0 is padding) with a mask, so `skills`/`skill_mask` are constant within an
episode (checked; a varying plan is an error) and contain no "current position". The
annotation IDs are 0 pick, 1 place, 2 push, 3 open/close (drawer, door), 4 turn (knob).
The training data has a per-frame `class` (see `data_process/libero/slice_indices`), but it
is not a policy input and no rollout equivalent is recorded. The current plan position is
therefore inferred from the executed gripper commands (`pred_chunk[:, :n_executed, 6] > 0`
is close; steps before the first inference are the dummy open action), debounced to
changes lasting at least 5 steps:

- pick ends when a grasp is completed (close event + 5 steps); place ends when the next
  release is completed. The closing/opening steps belong to the subtask they finish.
- a single-skill plan labels the whole episode.
- a non-grasp skill (push, open/close, turn) that is followed by more items ends with its
  own grasp-release cycle if the episode has more grasps than the remaining picks need
  (e.g. turning the knob); otherwise its boundary with the following pick is unobservable
  and those steps are `ambiguous` (e.g. "open the top drawer and put the bowl inside").
- a trailing non-grasp skill (e.g. close the drawer after placing) starts after the
  place's release. After the plan is consumed the last skill continues, except that an
  unexpected extra grasp makes the rest `ambiguous` (regrasp after a drop).
- if the episode runs out of events the current item continues to the end.

A window's subtask is the label at its observation step. Queries with an ambiguous label
are dropped from every condition, so all conditions use the same queries; ambiguous store
rows are candidates only in conditions a and b. This is a proxy for the high-level
subtask, not a recorded label.

**Conditions** (eligible store rows per query, always excluding same episode and same
initial-state group): a unconstrained; b other task (task string differs); c other
subtask (known, different skill ID); d (supplementary) other task with the same skill ID.
Per condition: feature/proprio kNN, exact random expectation and per-metric oracle over
the eligible rows. Feature standardization uses the whole store, so a filter only removes
candidates. One Bonferroni family covers conditions × features × k × components (vs
proprio, via `_compare_to_proprio`); a second family compares with random.

**Neighbour sources** (condition a): share of the k retrieved rows that come from same
task/same subtask, same task/other subtask, other task/same subtask, other task/other
subtask, unknown subtask; chance = candidate share; `oracle_motion_combined` ranks by the
combined motion distance (sum of the three components, each divided by its mean over all
eligible query-store pairs). Same-task share is computed from the task string alone, so
same-task rows with an unknown subtask count as same task. Excess over chance and same-task share minus the motion
oracle's share get cluster-bootstrap Bonferroni intervals.

**Cross-task recall.** tau = a quantile (0.5, 0.9) of the pooled combined distances from
each query to its 5 combined-motion-nearest unconstrained store rows. Relevant rows are
other-task rows with combined distance <= tau. recall@k = hits/|relevant|,
precision@k = hits/k (tie-weighted), over queries with at least one relevant row, ranked
within the other-task pool and within the unconstrained pool. Feature minus proprio gets
cluster-bootstrap Bonferroni intervals.
