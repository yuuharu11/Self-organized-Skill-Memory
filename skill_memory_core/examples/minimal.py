"""CPU-only end-to-end example; no robotics or checkpoint dependency."""
from __future__ import annotations

import numpy as np
import torch

from skill_memory_core.bridge import bridge_batch
from skill_memory_core.drafts import drafts_by_row_id
from skill_memory_core.key import SalientTokenKey, build_context, train_key
from skill_memory_core.memory import MotionMemory


def main():
    torch.set_num_threads(8)
    rng = np.random.default_rng(42)
    n, h, d = 16, 10, 32
    actions = rng.normal(size=(n, h, d)).astype(np.float32)
    tasks = np.repeat(np.arange(4, dtype=np.int32), 4)
    episodes = np.arange(n, dtype=np.int64)
    mem = MotionMemory(np.arange(n), actions, tasks, episodes)
    state = rng.normal(size=(n, 8)).astype(np.float32)
    context = build_context(state, actions[:, :, :7], np.ones(n), layout="as_trained")
    tokens = rng.normal(size=(n, 8, 16)).astype(np.float32)
    mask = np.ones((n, 8), bool)
    key = SalientTokenKey(dim=12, token_count=8, token_dim=16)
    batches = [tokens[:8], tokens[8:]], [context[:8], context[8:]], [mask[:8], mask[8:]]
    ab = [actions[:8, :, :7].reshape(8, -1), actions[8:, :, :7].reshape(8, -1)]
    tb = [tasks[:8], tasks[8:]]
    key, history = train_key(key, *batches, ab, tb, tau=100., epochs=2, device="cpu")

    with torch.no_grad():
        keys = key(torch.as_tensor(tokens), torch.as_tensor(context), torch.as_tensor(mask)).numpy()
    query_ids = np.arange(4, dtype=np.int64)
    memory_ids = np.arange(4, n, dtype=np.int64)
    # This small demonstration uses only task IDs to enforce cross-task shortlist.
    from skill_memory_core.key import coarse_search
    cand, sim = coarse_search(keys[query_ids], keys[memory_ids], query_ids, memory_ids,
                              tasks[query_ids], tasks[memory_ids], k=3)
    draft = drafts_by_row_id(actions, cand[:, 0])
    noise = np.zeros_like(draft)
    _, target, model_t = bridge_batch(draft, draft, noise,
        np.full(len(draft), .4, np.float32), np.full(len(draft), .5, np.float32))
    print({"rows": len(mem.row_ids), "key_updates": len(history), "top_candidates": cand.tolist(),
           "top_cosine": sim[:, 0].round(3).tolist(), "draft_shape": draft.shape,
           "bridge_target_shape": target.shape, "model_time": model_t.tolist()})


if __name__ == "__main__":
    main()
