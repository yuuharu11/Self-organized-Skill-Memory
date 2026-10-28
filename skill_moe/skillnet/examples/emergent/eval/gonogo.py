"""The first go/no-go, judged on the z actually used at inference.

Inference-time z = Prior z (observation only) and the memory z (top-k blend retrieved by
key from the memory). The encoder posterior z saw the action chunk and is reported only as
a reference; it never enters a criterion.

(i)  MI(Prior z clusters; env motion label) > label-shuffle p95, clearly above the
     untrained model (familywise bootstrap lower bound > 0) and above MI(Prior z; task)
     (both shuffle-adjusted, because the 40-class task label has a larger plug-in bias).
(ii) Retrieval restricted to other tasks closes >= 50% of the random-to-oracle gap
     (crosstask condition b_other_task) with an inference-time z.
(iii) Cross reconstruction: memory-blend z error < Prior z error (paired, familywise
     bootstrap lower bound of prior - memory_blend > 0).

Status per criterion: pass / fail / undetermined (missing baseline, too few groups, or
an interval that includes zero with a favourable point estimate). Overall: go if all pass,
no_go if any fails, otherwise undetermined.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

INFERENCE_SOURCES = ("prior", "memory")


@dataclass(frozen=True)
class GoNoGoCriteria:
    motion_label: str = "env/joint"
    gap_closed_min: float = 0.5
    crosstask_condition: str = "b_other_task"
    crosstask_k: str = "1"
    gap_components: tuple[str, ...] = ("eef_relative_position",)
    crosstask_sources: tuple[str, ...] = ("z_prior", "z_memory")
    crosstask_require: str = "any"     # "any" | "all" of crosstask_sources
    recon_mode: str = "memory_blend"


def _criterion_mi(mi: dict[str, Any], c: GoNoGoCriteria) -> dict[str, Any]:
    if mi is None or "prior" not in mi.get("table", {}):
        return {"status": "undetermined", "reason": "no prior-z MI result"}
    K = str(mi["primary_clusters"])
    entry = mi["table"]["prior"][K]
    motion, task = entry[c.motion_label], entry["task"]
    checks = {"above_shuffle_p95": motion["raw_bits"] > motion["shuffle_null_p95_bits"],
              "above_task_mi_shuffle_adjusted": motion["shuffle_adjusted_bits"] > task["shuffle_adjusted_bits"]}
    comp = [x for x in mi["comparisons_primary_k"]
            if x["source"] == "prior" and x["baseline"] == "untrained_prior" and x["label"] == c.motion_label]
    evidence = {"clusters": int(K), "motion_raw_bits": motion["raw_bits"],
                "motion_shuffle_p95_bits": motion["shuffle_null_p95_bits"],
                "motion_shuffle_adjusted_bits": motion["shuffle_adjusted_bits"],
                "task_shuffle_adjusted_bits": task["shuffle_adjusted_bits"]}
    if not comp:
        status = "fail" if not all(checks.values()) else "undetermined"
        return {"status": status, "checks": checks, "reason": "untrained-model z (z_untrained_prior_mean) missing",
                **evidence}
    ci = comp[0]["ci95_familywise"]
    evidence["minus_untrained_bits"] = comp[0]["estimate"]
    evidence["minus_untrained_ci95"] = ci
    if ci is None:
        checks["above_untrained"] = None
        status = "fail" if not all(v for v in checks.values() if v is not None) else "undetermined"
        return {"status": status, "checks": checks, "reason": "no bootstrap interval (too few groups)", **evidence}
    checks["above_untrained"] = ci[0] > 0
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks, **evidence}


def _criterion_crosstask(ct: dict[str, Any], c: GoNoGoCriteria) -> dict[str, Any]:
    if ct is None or "skipped" in ct:
        return {"status": "undetermined", "reason": (ct or {}).get("skipped", "no crosstask result")}
    cond = ct["item1_constrained_retrieval"].get(c.crosstask_condition)
    if cond is None:
        return {"status": "undetermined", "reason": f"condition {c.crosstask_condition} missing"}
    per = {}
    for src in c.crosstask_sources:
        name = f"feat__{src}"
        if name not in cond["gap_closed"]:
            continue
        gaps = {m: cond["gap_closed"][name][c.crosstask_k][m] for m in c.gap_components}
        per[src] = {"gap_closed": gaps, "pass": all(v >= c.gap_closed_min for v in gaps.values())}
    if not per:
        return {"status": "undetermined", "reason": "no inference-time z in crosstask features"}
    ok = [v["pass"] for v in per.values()]
    passed = any(ok) if c.crosstask_require == "any" else all(ok) and len(per) == len(c.crosstask_sources)
    return {"status": "pass" if passed else "fail", "by_source": per, "k": c.crosstask_k,
            "condition": c.crosstask_condition, "threshold": c.gap_closed_min, "require": c.crosstask_require}


def _criterion_recon(cr: dict[str, Any], c: GoNoGoCriteria) -> dict[str, Any]:
    if cr is None or "skipped" in cr:
        return {"status": "undetermined", "reason": (cr or {}).get("skipped", "no cross-reconstruction result")}
    row = [x for x in cr.get("improvement_over_prior", []) if x["mode"] == c.recon_mode]
    if not row:
        return {"status": "undetermined", "reason": f"mode {c.recon_mode} or prior missing"}
    est, ci = row[0]["estimate"], row[0]["ci95_familywise"]
    evidence = {"prior_mse": cr["mse"].get("prior"), f"{c.recon_mode}_mse": cr["mse"].get(c.recon_mode),
                "prior_minus_mode": est, "ci95_familywise": ci}
    if est <= 0:
        return {"status": "fail", **evidence}
    if ci is None:
        return {"status": "undetermined", "reason": "no bootstrap interval (too few groups)", **evidence}
    return {"status": "pass" if ci[0] > 0 else "undetermined",
            **({} if ci[0] > 0 else {"reason": "interval includes zero"}), **evidence}


def decide(mi: dict[str, Any] | None, crosstask: dict[str, Any] | None, crossrecon: dict[str, Any] | None,
           criteria: GoNoGoCriteria = GoNoGoCriteria()) -> dict[str, Any]:
    result = {"i_prior_mi": _criterion_mi(mi, criteria),
              "ii_crosstask_gap": _criterion_crosstask(crosstask, criteria),
              "iii_cross_reconstruction": _criterion_recon(crossrecon, criteria)}
    statuses = [v["status"] for v in result.values()]
    overall = "go" if all(s == "pass" for s in statuses) else "no_go" if "fail" in statuses else "undetermined"
    return {"overall": overall, "criteria": result, "settings": asdict(criteria),
            "judged_on": "inference-time z only (Prior z, memory blend z); posterior z is reference"}
