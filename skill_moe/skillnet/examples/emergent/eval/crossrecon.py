"""Go/no-go item 3: reconstruct a query's actions from a z retrieved from another episode.

The model is not imported. The caller passes ``reconstruct_fn(z, context) -> actions``:

* ``z``: float32 (B, Dz), one donor z per query row;
* ``context``: dict of length-B arrays describing the *query* rows: ``row_index`` (into the
  `ZRun` table), ``episode_id``, ``step``, ``proprio`` (and the run itself under ``run``),
  so a model wrapper can load the query observation it conditions on;
* returns (B, H, Da) actions for the query interval [step, step+H).

Error = masked MSE against the query's executed target actions (`z_target_actions`).
Donors always come from the store split and never share the query's episode or
initial-state group (optionally not its task). Donor modes:

``own`` query posterior z (upper reference); ``prior`` query prior z (no donor);
``memory_key`` donor posterior z of the key-nearest store row (realizable retrieval);
``oracle_motion`` donor whose observed future motion is nearest (G2 combined distance;
uses the query future -> diagnostic ceiling); ``random`` uniform eligible donors
(mean of `random_draws`); ``zero`` z = 0. ``extra_z`` adds per-row z matrices as further
modes, e.g. ``{"memory_blend": z_memory}`` = the top-k blend retrieved from the store-split
memory by key (what the policy would actually use). For every mode the paired error
difference to ``prior`` (prior minus mode; positive = mode better) is also bootstrapped.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.crosstask import _cluster_bootstrap, _pairwise_outcomes

from .data import ZRun

ReconstructFn = Callable[[np.ndarray, dict[str, Any]], np.ndarray]
MODES = ("own", "prior", "memory_key", "oracle_motion", "random", "zero")


def _masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    err = ((pred - target) ** 2).mean(axis=2)
    return (err * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)


def cross_reconstruction(run: ZRun, reconstruct_fn: ReconstructFn, query: np.ndarray, store: np.ndarray, *,
                         donor_source: str = "post", key: np.ndarray | None = None,
                         extra_z: dict[str, np.ndarray] | None = None,
                         other_task_only: bool = False, random_draws: int = 8, seed: int = 0,
                         batch_size: int = 256, bootstrap_samples: int = 2000) -> dict[str, Any]:
    if run.target_actions is None:
        return {"skipped": "no z_target_actions and no pred_chunk/n_executed to derive them"}
    q_rows = np.flatnonzero(query & run.usable)
    s_rows = np.flatnonzero(store & run.usable)
    if not len(q_rows) or not len(s_rows):
        raise ValueError("no usable query or store rows")
    if set(run.episode_id[q_rows]) & set(run.episode_id[s_rows]):
        raise AssertionError("query and donor episodes overlap")
    eligible = ((run.episode_id[q_rows][:, None] != run.episode_id[s_rows][None])
                & (run.init_state_id[q_rows][:, None] != run.init_state_id[s_rows][None]))
    if other_task_only:
        eligible &= run.task[q_rows][:, None] != run.task[s_rows][None]
    keep = eligible.any(axis=1)
    q_rows, eligible = q_rows[keep], eligible[keep]
    if not len(q_rows):
        raise ValueError("no query row has an eligible donor")
    z_donor = run.z[donor_source]
    rng = np.random.default_rng(seed)
    donors: dict[str, np.ndarray] = {}
    # Oracle: G2 combined motion distance (each component divided by its mean over eligible pairs).
    outcome = _pairwise_outcomes(run.motion[q_rows], run.motion[s_rows])
    scale = np.array([outcome[..., j][eligible].mean() for j in range(3)])
    combined = np.where(eligible, (outcome / np.maximum(scale, 1e-12)).sum(axis=2), np.inf)
    donors["oracle_motion"] = s_rows[combined.argmin(axis=1)]
    if key is not None:
        kq, ks = key[q_rows], key[s_rows]
        d = np.sqrt(np.maximum((kq ** 2).sum(1)[:, None] - 2 * kq @ ks.T + (ks ** 2).sum(1)[None], 0))
        donors["memory_key"] = s_rows[np.where(eligible, d, np.inf).argmin(axis=1)]
    random_idx = np.stack([s_rows[rng.choice(np.flatnonzero(e), size=random_draws)] for e in eligible])
    target, mask = run.target_actions[q_rows], run.target_mask[q_rows]

    def run_fn(z: np.ndarray) -> np.ndarray:
        out = []
        for start in range(0, len(q_rows), batch_size):
            sl = slice(start, start + batch_size)
            rows = q_rows[sl]
            context = {"row_index": rows, "episode_id": run.episode_id[rows], "step": run.step[rows],
                       "proprio": run.proprio[rows], "run": run}
            pred = np.asarray(reconstruct_fn(np.asarray(z[sl], dtype=np.float32), context), dtype=np.float64)
            if pred.shape != target[sl].shape:
                raise ValueError(f"reconstruct_fn returned {pred.shape}, expected {target[sl].shape}")
            out.append(_masked_mse(pred, target[sl], mask[sl]))
        return np.concatenate(out)

    errors = {"own": run_fn(run.z[donor_source][q_rows])}
    if "prior" in run.z and run.z["prior"].shape[1] == z_donor.shape[1]:
        errors["prior"] = run_fn(run.z["prior"][q_rows])
    for mode in ("memory_key", "oracle_motion"):
        if mode in donors:
            errors[mode] = run_fn(z_donor[donors[mode]])
    for mode, zz in (extra_z or {}).items():
        if not np.all(np.isfinite(zz[q_rows])):
            raise ValueError(f"extra z {mode} is not finite on query rows")
        errors[mode] = run_fn(zz[q_rows])
    draws = np.stack([run_fn(z_donor[random_idx[:, r]]) for r in range(random_draws)], axis=1)
    errors["random"] = draws.mean(axis=1)
    errors["zero"] = run_fn(np.zeros((len(q_rows), z_donor.shape[1])))
    # Paired improvements over random (positive = better than random) and gap to own.
    names = [m for m in errors if m != "random"]
    cols = [errors["random"] - errors[m] for m in names]
    prior_names = [m for m in errors if m not in ("prior", "own")] if "prior" in errors else []
    cols += [errors["prior"] - errors[m] for m in prior_names]
    boot = _cluster_bootstrap(np.stack(cols, axis=1), run.init_state_id[q_rows], seed + 7, bootstrap_samples)
    n_r = len(names)
    return {
        "donor_source": donor_source, "other_task_only": other_task_only,
        "n_queries": int(len(q_rows)), "n_query_init_states": int(len(np.unique(run.init_state_id[q_rows]))),
        "eligible_donors_per_query_mean": float(eligible.sum(1).mean()),
        "mse": {m: float(v.mean()) for m, v in errors.items()},
        "random_draw_mse": draws.mean(axis=0).tolist(),
        "improvement_over_random": [
            {"mode": m, "estimate": e, "ci95_familywise": ci}
            for m, e, ci in zip(names, boot["estimates"][:n_r], boot["intervals"][:n_r])],
        "improvement_over_prior": [
            {"mode": m, "estimate": e, "ci95_familywise": ci}
            for m, e, ci in zip(prior_names, boot["estimates"][n_r:], boot["intervals"][n_r:])],
        "bootstrap": {k: boot[k] for k in ("n_groups", "n_comparisons", "bootstrap_samples")},
        "donor_rows": {m: v.tolist() for m, v in donors.items()},
        "query_rows": q_rows.tolist(),
        "notes": "oracle_motion uses the query's observed future and is a ceiling, not a policy; "
                 "memory_key is the realizable retrieval. If own ~ random ~ zero the decoder ignores z.",
    }


def ridge_probe_decoder(run: ZRun, fit_rows: np.ndarray, *, source: str = "post", alpha: float = 1.0,
                        use_proprio: bool = True) -> ReconstructFn:
    """Stand-in for a model call (smoke tests only): linear map [z, proprio] -> flattened actions."""
    rows = np.flatnonzero(fit_rows & run.usable)
    z = run.z[source][rows]
    zm, zs = z.mean(0), z.std(0) + 1e-8
    pm, ps = run.proprio[rows].mean(0), run.proprio[rows].std(0) + 1e-8

    def features(zz, prop):
        parts = [(zz - zm) / zs] + ([(prop - pm) / ps] if use_proprio else [])
        return np.concatenate(parts + [np.ones((len(zz), 1))], axis=1)

    X = features(z, run.proprio[rows])
    Y = run.target_actions[rows].reshape(len(rows), -1)
    W = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)
    shape = run.target_actions.shape[1:]

    def fn(zz: np.ndarray, context: dict[str, Any]) -> np.ndarray:
        return (features(zz.astype(np.float64), context["proprio"]) @ W).reshape(len(zz), *shape)

    return fn
