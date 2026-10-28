"""Go/no-go item 2: cross-task retrieval with z, by reusing g2_retrieval.crosstask unchanged.

`crosstask.evaluate_crosstask` reads a run directory and compares every `feat__*` array
with proprio. We write a small *view* run whose NPZs hold only the arrays crosstask needs
plus `feat__z_<source>` (and optional extra features such as memory keys). Split,
leakage guards, subtask labels, kNN, random/oracle, recall and the Bonferroni bootstrap
are therefore exactly those of the G2 cross-task analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.crosstask import (
    CrossTaskConfig,
    evaluate_crosstask,
    write_report,
)
from skill_moe.skillnet.examples.memory_pilot.g2_retrieval.evaluate import EvaluationConfig

from .data import ZRun

VIEW_ARRAYS = ("step", "proprio", "env_eef_pos", "env_eef_quat", "env_gripper",
               "skills", "skill_mask", "pred_chunk", "n_executed", "segment_id")


def write_crosstask_view(run: ZRun, view_dir: str | Path, *, sources: tuple[str, ...] | None = None,
                         extra_features: Mapping[str, np.ndarray] | None = None) -> Path:
    """Write feat__z_<source> (+ extras, row-aligned with `run`) into a new view run dir.

    Rows that are not `usable` get zeros; this is safe only because crosstask drops every
    row without H future env steps, and the check below asserts that unusable rows have none.
    """
    view = Path(view_dir)
    (view / "episodes").mkdir(parents=True, exist_ok=False)
    bad = run.has_future & ~run.usable
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(f"row {run.episode_id[i]}@{run.step[i]} has a full env future but is not usable "
                         "(unexecuted or truncated z); crosstask would treat it as a valid window")
    features = {f"z_{s}": run.z[s] for s in (sources or sorted(run.z))}
    features.update(extra_features or {})
    for ep in run.episodes:
        rows = np.flatnonzero(run.episode_id == ep.episode_id)
        order = np.argsort(run.row[rows])
        rows = rows[order]
        payload = {k: ep.arrays[k] for k in VIEW_ARRAYS if k in ep.arrays}
        for name, mat in features.items():
            m = np.asarray(mat[rows], dtype=np.float32)
            m = np.where(run.usable[rows][:, None], m, 0.0)
            payload[f"feat__{name}"] = m
        np.savez(view / "episodes" / f"{ep.episode_id}.npz", **payload)
    manifest = dict(run.manifest)
    entries = manifest.get("episodes")
    if isinstance(entries, list):
        manifest["episodes"] = [{k: v for k, v in e.items() if k not in ("file", "path")} for e in entries]
    manifest["emergent_view_of"] = str(run.run_dir)
    (view / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return view


def evaluate_z_crosstask(run: ZRun, out_dir: str | Path, *, query_fraction: float = 0.25, seed: int = 0,
                         k_values: tuple[int, ...] = (1, 5), bootstrap_samples: int = 2000,
                         sources: tuple[str, ...] | None = None,
                         extra_features: Mapping[str, np.ndarray] | None = None,
                         plan_annotations: str | Path | None = None) -> dict[str, Any]:
    """Run crosstask on z with horizon = z interval H; writes view/ and crosstask_report.md."""
    out = Path(out_dir)
    missing = [k for k in ("skills", "skill_mask", "pred_chunk", "n_executed")
               if any(k not in ep.arrays for ep in run.episodes)]
    if missing:
        return {"skipped": f"crosstask needs NPZ arrays {missing} (see SCHEMA.md)"}
    view = write_crosstask_view(run, out / "crosstask_view", sources=sources, extra_features=extra_features)
    config = CrossTaskConfig(base=EvaluationConfig(horizon=run.horizon, query_fraction=query_fraction, seed=seed,
                                                   k_values=k_values, bootstrap_samples=bootstrap_samples))
    result = evaluate_crosstask(view, config, plan_annotations)
    write_report(result, out / "crosstask_report.md", run_dir=run.run_dir)
    return result


def summarize_crosstask(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gap closed (random -> oracle) per condition/method at each k, position component."""
    if "skipped" in result:
        return []
    rows = []
    for c, value in result["item1_constrained_retrieval"].items():
        for m, by_k in value["gap_closed"].items():
            for k, by_metric in by_k.items():
                decision = (result["item1_uncertainty_vs_random"]["by_condition"].get(c, {})
                            .get("proprio_knn" if m == "proprio" else m, {}).get("decision"))
                rows.append({"condition": c, "method": m, "k": k,
                             "gap_closed_position": by_metric["eef_relative_position"],
                             "gap_closed_rotation": by_metric["eef_relative_rotation_rad"],
                             "gap_closed_gripper": by_metric["gripper_relative"],
                             "vs_random_decision": decision})
    return rows
