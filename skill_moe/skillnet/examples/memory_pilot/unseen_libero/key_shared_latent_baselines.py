#!/usr/bin/env python3
"""CPU-only fixed-row random/hold/PI0REF/oracle baselines for KEY2-E."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from key_shared_latent import (
    _arm6_candidate_mse,
    _assert_hold_last_gate,
    _build_key2_queries,
    _candidate_window_actions,
    _summarize_split,
    _without_private,
    REPORT_RANKING_TAU,
    RANDOM20_BLEND_SEED,
)


def run(output: Path, *, vjepa_eval_cache: Path, run_root: Path,
        missing_root: Path, seen_stride5: Path, unseen_stride5: Path,
        parity_source: Path) -> dict:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    bank, _, _, _, _, learned, unseen, dev = _build_key2_queries(
        run_root, missing_root, seen_stride5, unseen_stride5, vjepa_eval_cache
    )
    if len(learned["key"]) != 621 or len(unseen["key"]) != 619:
        raise AssertionError("fixed learned/unseen evaluation rows changed")
    cactions = _candidate_window_actions(np.arange(len(bank["length"])), bank)
    hold_last_parity = _assert_hold_last_gate(learned, unseen)
    tau = REPORT_RANKING_TAU
    baselines = {}
    querywise = {}
    for name, split in (("learned", learned), ("unseen", unseen)):
        errors = _arm6_candidate_mse(split["actions"], cactions)
        best = errors.min(axis=1)
        hard = np.asarray(split['hard'], bool)
        querywise[name + '_all'] = float(best.mean())
        querywise[name + '_hard'] = float(best[hard].mean()) if hard.any() else None
        block = _summarize_split(name, np.zeros_like(errors), errors, split, bank,
            tau=tau, selected_blend={"k": 1, "temperature": 1.0},
            blend_by_k={"1": {"k": 1, "temperature": 1.0}}, reps=5000)
        block["rank_all"] = block["random_rank"]
        block["rank_hard"] = block["random_rank_hard"]
        block["actions_all"] = block["random5_actions_all"]
        block["actions_hard"] = block["random5_actions_hard"]
        baselines[name] = _without_private(block)
    smoke = json.loads(Path(parity_source).read_text())
    parity = smoke.get('parity', {})
    dim3 = parity.get('dim3_arm6_natural_dp', {})
    if (dim3.get('learned', {}).get('rows') != 621 or
        dim3.get('learned', {}).get('hard_rows') != 125 or
        dim3.get('unseen', {}).get('rows') != 619 or
        dim3.get('unseen', {}).get('hard_rows') != 134 or
        not parity.get('dim3_arm6_natural_dp_parity') or
        not parity.get('cmp3_natural_dct3_reference_matches')):
        raise AssertionError('parity source lacks full fixed-row DIM3/CMP3 smoke verification')
    result = {"status": "complete", "device": "cpu", "fixed_rows": {"learned": 621, "unseen": 619},
        "ranking_reference_tau": tau, "hold_last_parity": hold_last_parity,
        "random20_blend_seed": RANDOM20_BLEND_SEED,
        "random_seeds": [4101, 4102, 4103, 4104, 4105],
        "metrics": baselines,
        "reference_diagnostics": {"dim3_arm6_natural_dp": parity['dim3_arm6_natural_dp'],
            "querywise_best_arm6": querywise,
            "dim3_parity_verified_in_full_fixed_set_smoke": True,
            "parity_source": str(parity_source)},
        "rows_use_targets_only_for_scoring": True}
    (output / "baselines.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# KEY2-F CPU locked baselines", "", f"ranking tau={tau:.8g}; rows learned=621, unseen=619; random seeds=4101–4105", "",
             "|split|baseline|all MSE (gripper correct)|hard MSE (gripper correct)|", "|---|---|---:|---:|"]
    for name, block in baselines.items():
        for method in ("random5_actions_all", "hold_last", "zero", "random20_blend", "pi05", "oracle_playback_all"):
            if method == "random5_actions_all":
                allm = block[method].get("mse_gripper_truth")
                hardm = (block.get("random5_actions_hard") or {}).get("mse_gripper_truth")
            elif method == "oracle_playback_all":
                allm = block[method].get("mse_gripper_truth")
                hardm = (block.get("oracle_playback_hard") or {}).get("mse_gripper_truth")
            else:
                allm = (block.get("baselines") or {}).get(method, {}).get("mse_gripper_truth")
                hardm = (block.get("baselines_hard") or {}).get(method, {}).get("mse_gripper_truth")
            lines.append(f"|{name}|{method}|{allm}|{hardm}|")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vjepa-eval-cache", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, default=Path("/work/.skillnet-state/outputs/memory-pilot/pi05-memory40-20261104/run-01"))
    ap.add_argument("--missing-root", type=Path, default=Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/key-shared-01/attempt-02/prefix-missing"))
    ap.add_argument("--seen-stride5", type=Path, default=Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/seg5-cache"))
    ap.add_argument("--unseen-stride5", type=Path, default=Path("/work/.skillnet-state/outputs/memory-pilot/unseen-libero/seg-context-01/unseen-stride5"))
    ap.add_argument("--parity-source", type=Path, required=True,
        help="full-row smoke result with asserted DIM3/CMP3 oracle parity")
    args = ap.parse_args()
    result = run(args.output, vjepa_eval_cache=args.vjepa_eval_cache, run_root=args.run_root,
        missing_root=args.missing_root, seen_stride5=args.seen_stride5, unseen_stride5=args.unseen_stride5,
        parity_source=args.parity_source)
    print(json.dumps({"status": result["status"], "tau": result["teacher_tau"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
