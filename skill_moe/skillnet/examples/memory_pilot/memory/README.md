# Memory store API

This NumPy-only library stores **executed** motion segments independently from a policy.
It does not decide where or how retrieved values condition a model.

```python
from memory import MemoryMetadata, MemoryStore, namespace_init_state_id

store = MemoryStore(seed=7)
entry_id = store.append(
    MemoryMetadata(
        episode_id="ep-01",
        init_state_id=namespace_init_state_id("libero_spatial", 2, 0),
        task="pick up the block",
        segment_id="call-4",
        start_step=20,
        end_step=25,  # exclusive; five executed environment steps
        n_executed=5,
        representation_version="motion-v1",
        suite="libero_spatial",
    ),
    key=[0.1, 0.2],
    value=[1.0, 0.0, 0.5],
)
result = store.search(
    [0.0, 0.2], mode="topk_softmax", k=5, temperature=0.2,
    episode_id="query-episode",
    init_state_id=namespace_init_state_id("libero_spatial", 2, 1),
)
# result.candidate_ids, result.weights, result.composite_value,
# result.candidate_values (also result.z_collection), result.excluded
store.save("memory.npz")
loaded = MemoryStore.load("memory.npz")
```

`MemoryStore.from_recording(run_dir, key_feature=..., value_feature=...,
representation_version=...)` reads the gate recorder's manifest and episode NPZ files.
Feature arguments are NPZ keys such as `feat__L17_mean_t_last`; the same key may be
selected for both. To make z from executed rollout data, use
`value_source="pred_chunk_executed"` (clips each `pred_chunk` row to `[:n_executed]`) or
`value_source="env_trajectory"` (reconstructs relative end-effector position, rotation,
and gripper changes from `env_*` and `step`). The default `value_length` is manifest
`replan_steps`, falling back to the largest `n_executed` across the entire run. Both
sources are resampled once to this fixed length; observed rotations use quaternion
SLERP to avoid rotation-vector branch artifacts. Metadata records `value_source` and `value_length`.
The environment source requires a known environment time origin; format-v1 recorder data
uses the same offset inference as G2. See [SCHEMA.md](SCHEMA.md) for the index convention.
An imported row with `n_executed=0` is rejected. Search supports `none`, `random`,
`top1`, `topk_softmax`, `topk_uniform`, and `random_k`. Episode and namespaced initial-state matches
are excluded by default; retrieval requires both query IDs unless their respective guards
are explicitly disabled. Disabling guards is only suitable for a store already verified
disjoint from evaluation data. Use `namespace_init_state_id` for all initial-state IDs;
it matches G2's JSON tuple convention and preserves task ID zero. Bare IDs are rejected.
Format-v1 saved stores must be rebuilt; the current store format is v2.

Build the store from the memory split only, or supply exclusion sets covering all held-out
episodes and states. Otherwise other evaluation entries could enter retrieval and key
standardization even when the current query's IDs are excluded. Representation-version
filters are also available. All modes use identical eligibility filters and fit key
standardization on eligible stored keys only. `random` draws one eligible entry uniformly,
so it is a one-entry control for `top1`. `random_k` draws up to k eligible entries uniformly
without replacement, matching the cardinality of a kNN mixture. Its default `uniform`
weighting is equal weights. With `random_k_weighting="topk_softmax"`, softmax weights from
the nearest top-k distances (using the same temperature) are assigned in random draw order,
independent of sampled entries' distances. All modes expose the eligible candidate IDs and
use identical exclusions. kNN distance ties are broken uniformly with a separate seeded
generator so manifest order cannot systematically favor one value or change random draws.
Every retrieval appends candidates, weights,
distances, and exclusion reasons to `store.search_log`; set `search_log_path` to also append
JSONL records. Selected entries' `usage_count` increments and is persisted on save.
