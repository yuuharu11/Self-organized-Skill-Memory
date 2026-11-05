# situation_oracle: simulator state as a retrieval key

Plan registered before feature extraction and evaluation.

## Question and fixed populations

Can exact simulator situation retrieve a useful next 10-step motion for held-out LIBERO-10 (Long) query rows? Which of end-effector pose (E), nearest-object geometry (R), gripper/contact state (G), and executed history (H) matters? This estimates a privileged-input upper bound; it does not claim that a vision encoder can infer the same state.

Use `run-01` row IDs and source episode mapping from `data_manifest.json` (`source_episode_index`, task index/text, frame). Long query rows and hard mask are fixed from `local-key-02/predictions.npz`; expected counts are 548 and 63, with hard threshold 0.01549410422449. The primary candidate pool is the 3,857-row `raw-rows-noLong` list. An explicitly separate same-task/different-episode reference may retrieve only from Long-task episodes other than the query episode; it is descriptive retrieval only and is never used for fitting or weight selection. No exact query row or query episode may enter either candidate pool.

Training and weight selection use only non-Long train/dev rows from task IDs 10–39. Dev query episodes are those fixed in `local-key-02/protocol.json`; key weights are selected by active7 top-1 action MSE on Dev, with same-task exclusion. Long query features and actions are not consulted until the selected weight vector is frozen. Code asserts all fit/selection IDs have task ID >=10 and query episode IDs are disjoint from candidates.

## Source-state reconstruction and features

For every used row, resolve `(task_index, source_episode_index, frame_index)` to the corresponding original HDF5 demo. Reconstruct the per-frame observation by loading the HDF5 MuJoCo state into the matching LIBERO task simulator and calling forward. Require the stored 10x7 action window, after the run-01 normalization, to match the HDF5 actions at that exact frame within 1e-5 absolute tolerance. Any unmatched task/demo/frame or mismatch stops the run; do not repair with nearest-frame matching.

### Source-frame mapping clarification (before the full extraction)

The unsegmented LIBERO-40 RLDS source is explicitly `*_no_noops`; its frame index is not always the original HDF5 frame index. Resolve each source episode by its constant `source_episode_index` and task, then identify the unique HDF5 demo by an exact monotone subsequence match of the full raw action sequence (no nearest-action matching). Removed no-op frames create gaps in the HDF5 frame map. For each selected row, all ten active action vectors are checked against the mapped HDF5 frames at raw float32 tolerance 1e-6; independently, the source action window transformed with the run-01 quantile normalizer must match cached active7 actions within 1e-5. The simulator state is restored from the mapped original HDF5 frame. This changes only the frame-index interpretation needed to respect the already-specified no-op-filtered source; it does not change populations, features, weight selection, metrics, or decision criteria.

Feature blocks, with object names removed and nearest objects sorted only by geometric distance:
- E: end-effector world position and orientation.
- R: relative position and orientation of the nearest two movable object bodies.
- G: gripper opening plus binary contact/grasp indicator derived from simulator contacts.
- H: preceding realized 10x7 action window; zero and a validity bit if no preceding window exists.
- Required combinations: R+G, R+G+H, E+R+G+H. Also report component ablations E/R/G/H.

Standardization mean/std is fit from the non-Long memory rows only; near-zero standard deviations become one. Each block is separately standardized and normalized; candidate combinations use the predeclared small equal/half-weight grid, selected on non-Long Dev only. No object identity/name, task label, plan, or future action enters a key. Task identity is used only to construct the stated same-task reference and to assert the split.

## Evaluation

Queries: all 548 Long Eval rows and the fixed 63-row hard subset. Main candidates: raw 3,857 non-Long action rows. Reference candidates: same Long task, different episode only, for upper-bound context. Compare situation Top-1 and best action among situation Top-5; active7 action MSE, Recall@8 for candidate action MSE <= .01, xyz direction cosine, and gripper sign agreement. Also report hold-last, random candidate, the fixed local-key `wrist_lowerhalf+prev` ranking, and action-oracle baselines (.029/.035 as supplied). Compute episode-cluster bootstrap 95% CIs with 5,000 resamples for Top-1 MSE and paired differences. For k=8,32 nearest situations, report within-neighbor action dispersion as an empirical conditional-ambiguity floor.

Predeclared decisions:
- S1 passes only if the Dev-selected situation key has Long or hard paired Top-1 MSE difference below `wrist_lowerhalf+prev`, with the 95% CI upper endpoint <0.
- S2 passes only if the same frozen key beats hold-last on both all Long and hard subsets.
- Report E/R/G/H ablations to identify which state information matters.

## Checks and compute limits

Add tests that distinguish dataset row IDs from compact pool positions and validate candidate-index round trips, task/episode exclusions, HDF5 action-window alignment, feature block dimensions, and no-Long fit/selection. CPU only, no more than 8 Python threads for this run, `taskset -c 8-23`, `nice -n 10`; no GPU, training, or rollouts. Save arrays and logs only under `situation-oracle-01/`, never overwrite existing output.
