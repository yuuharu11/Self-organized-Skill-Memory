"""Four-key candidate union for the stage-3 saved-state branches.

The score definitions mirror score_audit/make_vlm_keys.py and keys.py:
grouped token MaxSim (all groups or wrist only), cosine of memory-standardized
codec prior b, and negative normalized action MSE from a pi0.5 prediction to
the recorded memory action.
"""
from __future__ import annotations

import numpy as np


KEY_ORDER = ('vlm_current', 'wrist_only', 'codec_prior_b', 'pi05_pred_to_memory_action')


def rank_scores(scores, row_ids, *, limit=None):
    """Descending score order with stable global row-id tie-breaking."""
    scores = np.asarray(scores, np.float32).reshape(-1)
    row_ids = np.asarray(row_ids, np.int64).reshape(-1)
    if scores.shape != row_ids.shape or not np.isfinite(scores).all():
        raise ValueError('scores and row_ids must be aligned finite vectors')
    order = np.lexsort((row_ids, -scores))
    if limit is not None:
        order = order[:int(limit)]
    return [{'memory_row': int(row_ids[i]), 'score': float(scores[i]), 'rank': int(j + 1)}
            for j, i in enumerate(order)]


def merge_key_rankings(rankings, *, per_key=2, target=8):
    """Take `per_key` distinct rows per key, filling overlaps by next rank.

    Once each source has contributed its quota, remaining slots are filled by
    cycling through sources in KEY_ORDER. Each returned row includes every
    source rank at which it appeared in the supplied ranking lists.
    """
    missing = [key for key in KEY_ORDER if key not in rankings]
    if missing:
        raise ValueError(f'missing retrieval rankings: {missing}')
    by_row = {}
    for key in KEY_ORDER:
        seen = set()
        for pos, item in enumerate(rankings[key]):
            row = int(item['memory_row'])
            rank = int(item.get('rank', pos + 1))
            if row in seen:
                continue
            seen.add(row)
            by_row.setdefault(row, {})[key] = {'rank': rank, 'score': float(item['score'])}

    selected = []
    selected_set = set()
    quota = {key: 0 for key in KEY_ORDER}
    cursors = {key: 0 for key in KEY_ORDER}
    # First ensure every key contributes its requested number of *unique* rows.
    for key in KEY_ORDER:
        source = rankings[key]
        while quota[key] < per_key and cursors[key] < len(source):
            item = source[cursors[key]]
            cursors[key] += 1
            row = int(item['memory_row'])
            if row in selected_set:
                continue
            selected.append(row)
            selected_set.add(row)
            quota[key] += 1
    # Defensive fill if a source had too few candidates or a malformed list.
    while len(selected) < target:
        progressed = False
        for key in KEY_ORDER:
            source = rankings[key]
            while cursors[key] < len(source):
                item = source[cursors[key]]
                cursors[key] += 1
                row = int(item['memory_row'])
                if row in selected_set:
                    continue
                selected.append(row)
                selected_set.add(row)
                progressed = True
                break
            if len(selected) >= target:
                break
        if not progressed:
            raise ValueError(f'only {len(selected)} unique candidates available; requested {target}')
    selected = selected[:target]
    return [{'memory_row': row, 'key_ranks': by_row.get(row, {})} for row in selected]


def standardized_unit_bank(values, mean=None, std=None):
    values = np.asarray(values, np.float32)
    mean = values.mean(0) if mean is None else np.asarray(mean, np.float32)
    std = values.std(0) if std is None else np.asarray(std, np.float32)
    std = np.where(std < 1e-5, 1., std).astype(np.float32)
    z = (values - mean) / std
    z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    return z.astype(np.float32), mean, std


def codec_cosine_scores(query_b, memory_b, mean=None, std=None):
    bank, mean, std = standardized_unit_bank(memory_b, mean, std)
    query, _, _ = standardized_unit_bank(np.asarray(query_b, np.float32).reshape(1, -1), mean, std)
    return bank @ query[0], mean, std


def action_key_scores(query_prediction, recorded_actions):
    """Exact score_audit.keys.generate_policy_keys `pi05_pred_to_memory_action`."""
    q = np.asarray(query_prediction, np.float32)
    a = np.asarray(recorded_actions, np.float32)
    if q.shape != (10, 7) or a.ndim != 3 or a.shape[1:] != (10, 32):
        raise ValueError(f'expected query [10,7] and actions [N,10,32], got {q.shape}, {a.shape}')
    return -np.mean((a[:, :, :7] - q[None]) ** 2, axis=(1, 2))


def raw_memory_draft(rows_actions, memory_row):
    """Exact memory_gap D-raw value: copy the selected normalized [10,32] row."""
    actions = np.asarray(rows_actions)
    draft = np.asarray(actions[int(memory_row)], np.float32).copy()
    if draft.shape != (10, 32):
        raise ValueError(f'D-raw expects one [10,32] row, got {draft.shape}')
    # This is intentionally not sliced to active-7: memory_gap passes all 32
    # stored normalized dimensions into sample_bridge.
    return draft
