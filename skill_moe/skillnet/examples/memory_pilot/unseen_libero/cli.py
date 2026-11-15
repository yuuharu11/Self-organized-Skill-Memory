"""Single entry point for the unseen_libero role subpackages.

Replaces the dozens of `python path/to/<name>.py` invocations with one
dispatcher::

    python cli.py list
    python cli.py run <module> [args ...]
    python -m skill_moe.skillnet.examples.memory_pilot.unseen_libero.cli run <module> [args ...]

<module> is either a short script name (e.g. ``key_query_probe``) or a
dotted ``<subpackage>.<script>`` path. Extra args are forwarded verbatim to
the target script's own argparse, so every script keeps its CLI unchanged.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODULES = {
    # queries/
    "image_query": "queries.image_query",
    "image_query_gate": "queries.image_query_gate",
    "image_query_metrics": "queries.image_query_metrics",
    "image_query_recheck": "queries.image_query_recheck",
    "image_query_robust": "queries.image_query_robust",
    "k2_unseen_layers": "queries.k2_unseen_layers",
    "k3_query": "queries.k3_query",
    "k4_query": "queries.k4_query",
    "key_query_k2": "queries.key_query_k2",
    "key_query_probe": "queries.key_query_probe",
    "fast_query": "queries.fast_query",
    "spatial_readout_query": "queries.spatial_readout_query",
    "scale_only_query": "queries.scale_only_query",
    "x_attn_query": "queries.x_attn_query",
    "wam_aux_query": "queries.wam_aux_query",
    "partial_denoise_query": "queries.partial_denoise_query",
    "unit_retrieval": "queries.unit_retrieval",
    "pi05_reference": "queries.pi05_reference",
    "l90x_rows": "queries.l90x_rows",
    "prepare_oracle": "queries.prepare_oracle",
    # experts/
    "expert_refine": "experts.expert_refine",
    "skill_expert_closed_loop": "experts.skill_expert_closed_loop",
    "skill_expert_denoise": "experts.skill_expert_denoise",
    # closed_loop/
    "run": "closed_loop.run",
    "worker": "closed_loop.worker",
    "serve": "closed_loop.serve",
    "memory_closed_loop": "closed_loop.memory_closed_loop",
    "magnitude_closed_loop": "closed_loop.magnitude_closed_loop",
    "progress_closed_loop": "closed_loop.progress_closed_loop",
    "seg_context_server": "closed_loop.seg_context_server",
    "seg_context_rollout": "closed_loop.seg_context_rollout",
    "theta_flow": "closed_loop.theta_flow",
    # probes/
    "key_split_ablation": "probes.key_split_ablation",
    "gripper_split": "probes.gripper_split",
    "scene_invariance": "probes.scene_invariance",
    "scene_invariance_v2": "probes.scene_invariance_v2",
    "magnitude_bounds": "probes.magnitude_bounds",
    "magnitude_orientation": "probes.magnitude_orientation",
    "theta_regression": "probes.theta_regression",
    "theta_regression_lora": "probes.theta_regression_lora",
    "lora_theta_on_selected": "probes.lora_theta_on_selected",
    "phase_probe": "probes.phase_probe",
    "orientation_fix": "probes.orientation_fix",
    "blend_confidence": "probes.blend_confidence",
    "cluster_pools": "probes.cluster_pools",
    "seg_context_extract": "probes.seg_context_extract",
    "seg_context_retrieval": "probes.seg_context_retrieval",
    "seg_context_run": "probes.seg_context_run",
    "latent_probe_data": "probes.latent_probe_data",
    "latent_probe_features": "probes.latent_probe_features",
    "latent_probe_gate": "probes.latent_probe_gate",
    "latent_probe_readout": "probes.latent_probe_readout",
    "noro_closed_loop": "probes.noro_closed_loop",
    "noro_run": "probes.noro_run",
    "eef_frame_units": "probes.eef_frame_units",
    "lib_closed": "probes.lib_closed",
    # analysis/
    "compressed_denoise": "analysis.compressed_denoise",
    "compression_compare": "analysis.compression_compare",
    "compression_matched": "analysis.compression_matched",
    "compression_per_axis": "analysis.compression_per_axis",
    "compression_per_axis_arm": "analysis.compression_per_axis_arm",
    "compression_per_axis_segmentation": "analysis.compression_per_axis_segmentation",
    "compression_seen_cuts": "analysis.compression_seen_cuts",
    "dct_coefficients_eval": "analysis.dct_coefficients_eval",
    "dct_skill_draft": "analysis.dct_skill_draft",
    "dct_skill_draft_eval": "analysis.dct_skill_draft_eval",
    "dct_tolerance_merge": "analysis.dct_tolerance_merge",
    "denoise_eval": "analysis.denoise_eval",
    "denoise_oracle_adaptive": "analysis.denoise_oracle_adaptive",
    "denoise_oracle_drafts": "analysis.denoise_oracle_drafts",
    "draft_reconstruction": "analysis.draft_reconstruction",
    "draft_reconstruction_bundles": "analysis.draft_reconstruction_bundles",
    "eval_transfer_diagnostic": "analysis.eval_transfer_diagnostic",
    "motion_core_eval": "analysis.motion_core_eval",
    "optimal_units_eval": "analysis.optimal_units_eval",
    "stored_units_eval": "analysis.stored_units_eval",
    "oracle_experiment": "analysis.oracle_experiment",
    "oracle_repr": "analysis.oracle_repr",
    "segment_demo": "analysis.segment_demo",
    "variable_length_error": "analysis.variable_length_error",
    "variable_memory_oracle": "analysis.variable_memory_oracle",
    "variable_shape_memory": "analysis.variable_shape_memory",
    "low_freq_denoise": "analysis.low_freq_denoise",
}

BASE = "skill_moe.skillnet.examples.memory_pilot.unseen_libero"


def resolve(name: str) -> str:
    if "." in name:
        return f"{BASE}.{name}"
    hits = [v for k, v in MODULES.items() if k == name]
    if not hits:
        raise KeyError(f"unknown module {name!r}; see `cli.py list`")
    return f"{BASE}.{hits[0]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print short name -> dotted module lines")
    run_p = sub.add_parser("run", help="run a module, forwarding extra args")
    run_p.add_argument("module", help="short name or <subpackage>.<script>")
    run_p.add_argument("args", nargs=argparse.REMAINDER, help="forwarded args")
    ns = parser.parse_args(argv)
    if ns.cmd == "list":
        for short in sorted(MODULES):
            print(f"{short} -> {BASE}.{MODULES[short]}")
        return 0
    target = resolve(ns.module)
    sys.argv = [target] + ns.args
    runpy.run_module(target, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
