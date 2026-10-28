"""Run every emergent-z evaluation on one recording and write results.json + report.md.

    python -m skill_moe.skillnet.examples.emergent.eval.pipeline <run_dir> [--horizon H]
        [--alias z_post_mean=feat__L09_mean_t_last ...] [--output-dir DIR]

Without a model hook, cross reconstruction uses `ridge_probe_decoder` (a linear stand-in,
labelled as such); pass `reconstruct_fn` to `run_all` for the real model.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .crossrecon import ReconstructFn, cross_reconstruction, ridge_probe_decoder
from .data import ZRun, load_z_run, split_rows
from .gonogo import GoNoGoCriteria, decide
from .memory import VLM_BLOCKS, EmergentMemory, KeyEncoder, MemoryConfig
from .metrics import MIConfig, evaluate_mi
from .retrieval import evaluate_z_crosstask, summarize_crosstask


def key_inputs(run: ZRun, rows: np.ndarray) -> dict[str, np.ndarray]:
    data = {name: run.keys[name][rows] for name in run.keys}
    data["proprio"] = run.proprio[rows]
    return data


def build_memory(run: ZRun, store: np.ndarray, *, include_language: bool, n_components: int | float = 32,
                 config: MemoryConfig = MemoryConfig(), exclude_rows: np.ndarray | None = None,
                 log_path: Path | None = None) -> EmergentMemory:
    """Fit the fixed PCA on store-split usable rows, seed them all (posterior z), calibrate tau."""
    rows = np.flatnonzero(store & run.usable & (True if exclude_rows is None else ~exclude_rows))
    encoder = KeyEncoder.fit(key_inputs(run, rows), include_language=include_language, n_components=n_components)
    mem = EmergentMemory(encoder, config, horizon=run.horizon, log_path=log_path)
    keys = encoder.transform(key_inputs(run, rows))
    for i, key in zip(rows, keys):
        mem.add(key, run.z["post"][i], episode_id=run.episode_id[i], init_state_id=run.init_state_id[i],
                task=run.task[i], suite=run.suite[i], step=int(run.step[i]))
    mem.calibrate()
    return mem


def memory_z(run: ZRun, mem: EmergentMemory) -> tuple[np.ndarray, np.ndarray]:
    """Top-k blend for every usable row (leave-episode/state-out); returns (z, PCA keys)."""
    z = np.full_like(run.z["post"], np.nan)
    keys = np.zeros((len(run), mem.encoder.components.shape[0]))
    rows = np.flatnonzero(run.usable)
    keys[rows] = mem.encoder.transform(key_inputs(run, rows))
    for i in rows:
        r = mem.retrieve(keys[i], episode_id=run.episode_id[i], init_state_id=run.init_state_id[i],
                         count_usage=False)
        if r.z is not None:
            z[i] = r.z
    return z, keys


ELIGIBLE_REASONS = ("near_existing", "added_novel", "added_no_threshold_yet")


def replay(run: ZRun, mem: EmergentMemory, rows: np.ndarray, *, seen_suites: set[str] | None = None,
           rng_seed: int = 0) -> dict[str, Any]:
    """Stream rows (temporal order per episode) through search-then-add; summarize the log."""
    start = len(mem.log)
    order = rows[np.lexsort((run.step[rows], run.episode_id[rows]))]
    z_err, z_err_random = [], []
    rng = np.random.default_rng(rng_seed)
    all_z = np.stack([e.value for e in mem.store.entries])
    for i in order:
        key = mem.encoder.transform(key_inputs(run, np.array([i])))[0]
        z = run.z["post"][i] if run.usable[i] else None
        r, _ = mem.observe(key, z, executed=bool(run.executed[i]), interval_len=int(run.interval_len[i]),
                           episode_id=run.episode_id[i], init_state_id=run.init_state_id[i], task=run.task[i],
                           suite=run.suite[i], step=int(run.step[i]),
                           extra_log={"seen_suite": None if seen_suites is None else run.suite[i] in seen_suites})
        if run.usable[i] and r.z is not None:
            z_err.append(float(np.linalg.norm(r.z - run.z["post"][i])))
            pick = rng.choice(len(all_z), size=len(r.candidate_ids), replace=False)
            z_err_random.append(float(np.linalg.norm(r.weights @ all_z[pick] - run.z["post"][i])))
    log = mem.log[start:]
    reasons = Counter(e["reason"] for e in log)
    eligible = [e for e in log if e["reason"] in ELIGIBLE_REASONS]
    out = {"n_observed": len(log), "n_added": int(sum(e["added"] for e in log)),
           "add_rate": float(np.mean([e["added"] for e in log])) if log else 0.0,
           "add_rate_eligible": float(np.mean([e["added"] for e in eligible])) if eligible else 0.0,
           "n_eligible": len(eligible), "reasons": dict(reasons), "tau": mem.tau, "d_ref": mem.d_ref, "memory_size_end": len(mem.store),
           "z_blend_error_mean": float(np.mean(z_err)) if z_err else None,
           "z_random_blend_error_mean": float(np.mean(z_err_random)) if z_err_random else None,
           "unexecuted_or_truncated_added": int(sum(e["added"] and e["reason"] not in ("added_novel",
                                                                                       "added_no_threshold_yet")
                                                    for e in log))}
    if seen_suites is not None:
        for flag, name in ((True, "seen_suite"), (False, "unseen_suite")):
            sel = [e for e in eligible if e["seen_suite"] is flag]
            out[f"add_rate_{name}"] = float(np.mean([e["added"] for e in sel])) if sel else None
            out[f"n_{name}"] = len(sel)
    return out


def run_all(run_dir: str | Path, out_dir: str | Path, *, horizon: int | None = None,
            aliases: dict[str, str] | None = None, query_fraction: float = 0.25, seed: int = 0,
            mi_config: MIConfig = MIConfig(), memory_config: MemoryConfig = MemoryConfig(),
            n_components: int | float = 32, reconstruct_fn: ReconstructFn | None = None,
            bootstrap_samples: int = 2000, criteria: GoNoGoCriteria = GoNoGoCriteria(),
            label: str = "") -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    run = load_z_run(run_dir, horizon=horizon, aliases=aliases)
    query, store = split_rows(run, query_fraction=query_fraction, seed=seed)
    results: dict[str, Any] = {"run_dir": str(run_dir), "label": label, "horizon": run.horizon,
                               "derived": run.derived, "query_fraction": query_fraction, "seed": seed,
                               "counts": {"rows": len(run), "usable": int(run.usable.sum()),
                                          "query_usable": int((query & run.usable).sum()),
                                          "store_usable": int((store & run.usable).sum()),
                                          "episodes": len(run.episodes)},
                               "memory_quantile": memory_config.threshold_quantile}
    # ---- memory: keys, memory z, acquisition replay -----------------------
    variants = [lang for lang in (False, True) if VLM_BLOCKS[lang] in run.keys]
    extra_features, memory_results = {}, {}
    # For memory z every query excludes its own episode and initial-state group (G2 rule).
    eval_config = MemoryConfig(**{**asdict(memory_config), "exclude_same_init_state": True})
    for lang in variants:
        tag = "lang" if lang else "nolang"
        mem = build_memory(run, store, include_language=lang, n_components=n_components, config=eval_config)
        zm, keys = memory_z(run, mem)
        name = "memory" if tag == ("nolang" if False in variants else "lang") else f"memory_{tag}"
        run.z[name] = zm
        run.z_logvar[name] = np.full_like(zm, np.nan)
        extra_features[f"key_{tag}"] = keys
        rep_mem = build_memory(run, store, include_language=lang, n_components=n_components, config=memory_config,
                               log_path=out / f"memory_log_{tag}.jsonl")
        entry = {"z_name": name, "pca_components": int(mem.encoder.components.shape[0]),
                 "pca_explained_variance": float(mem.encoder.explained_variance_ratio.sum()),
                 "seed_entries": len(rep_mem.store), "tau": rep_mem.tau, "d_ref": rep_mem.d_ref,
                 "replay_query_split": replay(run, rep_mem, np.flatnonzero(query), rng_seed=seed)}
        suites = sorted(set(run.suite))
        if len(suites) >= 2:
            held = suites[-1]
            held_rows = run.suite == held
            hs_mem = build_memory(run, store, include_language=lang, n_components=n_components,
                                  config=memory_config, exclude_rows=held_rows)
            entry["replay_held_out_suite"] = {"held_out_suite": held, **replay(
                run, hs_mem, np.flatnonzero(query), seen_suites=set(suites) - {held}, rng_seed=seed)}
        rep_mem.save(out / f"memory_{tag}")
        memory_results[tag] = entry
    results["memory"] = memory_results or {"skipped": "no key__vlm_img_* arrays"}
    # ---- (1) MI ------------------------------------------------------------
    results["mi"] = evaluate_mi(run, query, store, mi_config)
    # ---- (2) cross-task ------------------------------------------------------
    ct = evaluate_z_crosstask(run, out, query_fraction=query_fraction, seed=seed, bootstrap_samples=bootstrap_samples,
                              extra_features=extra_features)
    results["crosstask_summary"] = summarize_crosstask(ct)
    (out / "crosstask_results.json").write_text(json.dumps(ct, indent=1), encoding="utf-8")
    # ---- (3) cross reconstruction ------------------------------------------
    decoder_note = "caller-supplied model"
    if reconstruct_fn is None and run.target_actions is not None:
        reconstruct_fn = ridge_probe_decoder(run, store)
        decoder_note = "ridge probe [z_post, proprio] -> actions fitted on store split (stand-in, NOT the model)"
    if reconstruct_fn is not None:
        main_key = extra_features.get("key_nolang", extra_features.get("key_lang"))
        extra = {"memory_blend": run.z["memory"]} if "memory" in run.z else {}
        cr = cross_reconstruction(run, reconstruct_fn, query, store, key=main_key, extra_z=extra, seed=seed,
                                  bootstrap_samples=bootstrap_samples)
        cr.pop("donor_rows", None)
        cr.pop("query_rows", None)
        cr["decoder"] = decoder_note
        results["cross_reconstruction"] = cr
    else:
        results["cross_reconstruction"] = {"skipped": "no target actions"}
    results["gonogo"] = decide(results["mi"], ct, results["cross_reconstruction"], criteria)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    write_report(results, out / "report.md")
    return results


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}g}" if isinstance(x, float) else str(x)


def _ci(ci):
    return "n/a" if not ci else f"[{ci[0]:.3g}, {ci[1]:.3g}]"


def write_report(results: dict[str, Any], path: Path, preamble: str = "") -> None:
    g = results["gonogo"]
    c = results["counts"]
    lines = [f"# Emergent z evaluation {results.get('label', '')}".rstrip(), "",
             f"Recording: `{results['run_dir']}`; H={results['horizon']}; split seed {results['seed']} "
             f"(query fraction {results['query_fraction']}, whole initial-state groups).", "",
             f"Rows {c['rows']} (usable {c['usable']}; query {c['query_usable']}, store {c['store_usable']}), "
             f"episodes {c['episodes']}.", ""]
    if preamble:
        lines += [preamble, ""]
    if results["derived"]:
        lines += ["Substitutions: " + "; ".join(f"`{k}` = {v}" for k, v in results["derived"].items()), ""]
    lines += ["## Go/no-go (inference-time z only: Prior z, memory blend z)", "",
              f"Overall: **{g['overall']}**", "", "| Criterion | Status | Evidence |", "|---|---|---|"]
    for name, v in g["criteria"].items():
        ev = {k: val for k, val in v.items() if k != "status"}
        lines.append(f"| {name} | {v['status']} | `{json.dumps(ev, default=float)[:400]}` |")
    lines += ["", "Posterior z (Enc saw the action chunk) is a reference only and never enters a criterion.", ""]
    mi = results["mi"]
    K = str(mi["primary_clusters"])
    lines += ["## 1. MI between clustered z and motion (query split)", "",
              f"k-means fitted on store rows, MI on {mi['n_query_rows']} query rows / "
              f"{mi['n_query_init_states']} groups / {mi['n_query_tasks']} tasks. K={K}. "
              "Bits: raw (shuffle-adjusted). `env/*` = observed eef trajectory; `action/*` = the encoder's "
              "input actions (triviality reference).", ""]
    labels = [k for k in next(iter(mi["table"].values()))[K] if "/" in k and "given" not in k] + ["task"]
    lines += ["| z | role | " + " | ".join(labels) + " | I(motion;z\\|task) | I(task;z\\|motion) |",
              "|---|---|" + "---:|" * (len(labels) + 2)]
    for name, by_k in mi["table"].items():
        e = by_k[K]
        role = ("inference" if name in ("prior", "memory") or name.startswith("memory_")
                else "reference (saw actions)" if name == "post" else "baseline")
        cells = [f"{e[l]['raw_bits']:.3f} ({e[l]['shuffle_adjusted_bits']:.3f})" for l in labels]
        lines.append(f"| {name} | {role} | " + " | ".join(cells) + f" | {e['env/joint_given_task']['shuffle_adjusted_bits']:.3f}"
                     f" | {e['task_given_env/joint']['shuffle_adjusted_bits']:.3f} |")
    lines += ["", "Differences at the primary K (bits, familywise 95% whole-group bootstrap):", "",
              "| z | baseline | label | diff | interval |", "|---|---|---|---:|---|"]
    for x in mi["comparisons_primary_k"]:
        lines.append(f"| {x['source']} | {x['baseline']} | {x['label']} | {x['estimate']:.3f} | {_ci(x['ci95_familywise'])} |")
    p = mi["posterior"]
    lines += ["", f"Posterior: KL to N(0,I) {p['kl_to_standard_normal_nats_per_row']:.3g} nats/row, "
              f"active dims {p['active_dims_kl_gt_0.01']}/{p['z_dim']}"
              + (f", KL(post||prior) {p['kl_post_to_prior_nats_per_row']:.3g}" if "kl_post_to_prior_nats_per_row" in p else "")
              + (f", prior-post distance / shuffled {p['prior_to_post_distance_ratio_vs_shuffled']:.3f}"
                 if "prior_to_post_distance_ratio_vs_shuffled" in p else "") + ".", "", mi["notes"], ""]
    lines += ["## 2. Cross-task retrieval (G2 crosstask on z; details in crosstask_report.md)", "",
              "| condition | method | k | gap closed pos | rot | grip | vs random |", "|---|---|---:|---:|---:|---:|---|"]
    for r in results["crosstask_summary"]:
        if r["condition"] in ("a_unconstrained", "b_other_task"):
            lines.append(f"| {r['condition']} | {r['method']} | {r['k']} | {r['gap_closed_position']:.2f} | "
                         f"{r['gap_closed_rotation']:.2f} | {r['gap_closed_gripper']:.2f} | {r['vs_random_decision']} |")
    cr = results["cross_reconstruction"]
    lines += ["", "## 3. Cross reconstruction", ""]
    if "skipped" in cr:
        lines.append(f"Skipped: {cr['skipped']}")
    else:
        lines += [f"Decoder: {cr['decoder']}. Queries {cr['n_queries']} ({cr['n_query_init_states']} groups), "
                  f"donors per query {cr['eligible_donors_per_query_mean']:.0f}.", "",
                  "| mode | MSE | random - mode | interval | prior - mode | interval |", "|---|---:|---:|---|---:|---|"]
        imp = {x["mode"]: x for x in cr["improvement_over_random"]}
        pri = {x["mode"]: x for x in cr["improvement_over_prior"]}
        for m, v in cr["mse"].items():
            a, b = imp.get(m), pri.get(m)
            lines.append(f"| {m} | {v:.4g} | {_f(a['estimate'] if a else None)} | {_ci(a['ci95_familywise']) if a else ''} | "
                         f"{_f(b['estimate'] if b else None)} | {_ci(b['ci95_familywise']) if b else ''} |")
        lines += ["", cr["notes"]]
    lines += ["", "## 4. Memory (fixed PCA key, search-then-add, relative threshold)", ""]
    for tag, m in results["memory"].items():
        if not isinstance(m, dict):
            continue
        rq = m["replay_query_split"]
        lines += [f"### key: {tag}", "",
                  f"PCA {m['pca_components']} comps ({m['pca_explained_variance']:.2f} variance); seed entries "
                  f"{m['seed_entries']}; tau = q{results.get('memory_quantile', '')} of within-memory NN distance = "
                  f"{_f(m['tau'])} (median {_f(m['d_ref'])}).", "",
                  f"Query-split replay: observed {rq['n_observed']}, added {rq['n_added']} (rate among executed full-interval rows {rq['add_rate_eligible']:.3f}, "
                  f"expected about {1 - results['memory_quantile']:.2f} for in-distribution rows); "
                  f"reasons {rq['reasons']}; unexecuted/truncated added: {rq['unexecuted_or_truncated_added']}. "
                  f"Blend z error vs own posterior {_f(rq['z_blend_error_mean'])} (random same-size blend "
                  f"{_f(rq['z_random_blend_error_mean'])}).", ""]
        if "replay_held_out_suite" in m:
            h = m["replay_held_out_suite"]
            lines += [f"Held-out suite `{h['held_out_suite']}` (absent from seed memory): add rate unseen "
                      f"{_f(h['add_rate_unseen_suite'])} (n={h['n_unseen_suite']}) vs seen {_f(h['add_rate_seen_suite'])} "
                      f"(n={h['n_seen_suite']}).", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--alias", action="append", default=[], help="schema_key=npz_key")
    ap.add_argument("--query-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--pca-components", type=int, default=32)
    ap.add_argument("--label", default="")
    ap.add_argument("--output-dir", type=Path)
    a = ap.parse_args()
    out = a.output_dir or Path("/work/.skillnet-state/outputs/memory-pilot") / (
        "emergent-eval-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    aliases = dict(x.split("=", 1) for x in a.alias)
    run_all(a.run_dir, out, horizon=a.horizon, aliases=aliases, query_fraction=a.query_fraction, seed=a.seed,
            n_components=a.pca_components, bootstrap_samples=a.bootstrap_samples, label=a.label)
    print(out)


if __name__ == "__main__":
    main()
