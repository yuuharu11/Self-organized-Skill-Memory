"""Generate descriptive tables/figures from the frozen diagnostic collection.

Expert IDs are arbitrary learned routes, not named skills. Only baseline
queries enter utilization plots; repeated branches are not extra samples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def routing_summary(records):
    counts = np.zeros((2, 18, 4), np.int64)
    norm_sum = np.zeros((2, 18, 2), np.float64)
    exposures = np.zeros(2, np.int64)
    for r in records:
        placement = int(bool(r["stage_truth"][1]))
        d = r["response"]["diagnostics"]
        routes = np.asarray(d["router_top1"])[:, :5]
        selected = np.asarray(d["selected_contribution_norm"])[:, :5]
        shared = np.asarray(d["shared_output_norm"])[:, :5]
        if routes.shape != (18, 5) or not np.isin(routes, [0, 1, 2, 3]).all():
            raise ValueError("Expected18 layers x5 executed action-token routes")
        for layer in range(18):
            counts[placement, layer] += np.bincount(routes[layer], minlength=4)
        norm_sum[placement, :, 0] += selected.sum(axis=1)
        norm_sum[placement, :, 1] += shared.sum(axis=1)
        exposures[placement] += 5
    return counts, norm_sum, exposures


def repeat_summary(original, repeated):
    a = {r["step"]: r for r in original}
    b = {r["step"]: r for r in repeated}
    steps = sorted(a.keys() & b.keys())
    return {"common_queries": len(steps), "queries_original": len(a), "queries_repeat": len(b),
            "all_policy_images_identical": all(a[s]["image_sha256"] == b[s]["image_sha256"] for s in steps),
            "max_action_difference": max((float(np.max(np.abs(np.asarray(a[s]["response"]["actions"]) -
                                            np.asarray(b[s]["response"]["actions"])))) for s in steps), default=None)}


def analyze(root):
    baseline = []
    repeats = {}
    outcomes = []
    interventions = []
    for case in sorted(root.glob("task*-init*")):
        rows = read_jsonl(case / "baseline/queries.jsonl")
        baseline.extend(rows)
        for branch in ("baseline", "original", "remaining_instruction", "original_repeat", "released_original", "released_remaining", "released_original_repeat"):
            p = case / branch / "result.json"
            if p.exists():
                outcomes.append({"case": case.name, "branch": branch, **json.loads(p.read_text())})
        if (case / "original_repeat/result.json").exists():
            repeats[case.name] = repeat_summary(read_jsonl(case / "original/queries.jsonl"),
                                                read_jsonl(case / "original_repeat/queries.jsonl"))
        if (case / "released_original_repeat/result.json").exists():
            repeats[case.name + "/released_public"] = repeat_summary(read_jsonl(case / "released_original/queries.jsonl"),
                                                read_jsonl(case / "released_original_repeat/queries.jsonl"))
        for path in (case / "baseline/counterfactuals.jsonl", case / "baseline/snapshot/counterfactuals.jsonl"):
            for r in read_jsonl(path):
                interventions.append({"case": case.name, "snapshot": "placement" if path.parent.name == "snapshot" else "initial",
                                      **{k: v for k, v in r.items() if k != "response"}})
    counts, norms, exposures = routing_summary(baseline)
    derived = root / "analysis"
    derived.mkdir(exist_ok=True)
    with (derived / "expert_utilization.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["placement_true", "layer_zero_based", "expert_id", "tokens", "phase_tokens_per_layer", "fraction"])
        for p in range(2):
            for layer in range(18):
                for e in range(4):
                    writer.writerow([p, layer, e, counts[p, layer, e], exposures[p],
                                     counts[p, layer, e] / exposures[p] if exposures[p] else ""])
    survey_path = root / "reference_initial_survey/summary.json"
    survey = json.loads(survey_path.read_text()) if survey_path.exists() else []
    with (derived / "initial_skill_sensitivity.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["task_number", "condition", "public_translation_command_rms", "public_rotation_command_rms",
                         "public_gripper_command_rms", "instrumented_route_change_fraction", "capture_action_max_abs"])
        for row in survey:
            for condition in row["conditions"]:
                writer.writerow([row["task_number"], condition["condition"], *condition["action_rms_by_family"],
                                 condition["instrumented_route_change_fraction"], condition["capture_action_max_abs"]])
    with (derived / "branch_outcomes.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "branch", "success", "start_step", "end_step", "budget", "behavior_source"])
        for row in outcomes:
            writer.writerow([row["case"], row["branch"], row["success"], row["start_step"], row["end_step"], row["budget"],
                             "untouched_public_server" if row["branch"].startswith("released_") else "instrumented"])
    result = {"collection_complete": (root / "COLLECTION_DONE").exists(),
              "released_followup_complete": (root / "RELEASED_FOLLOWUP_DONE").exists(),
              "reference_survey_complete": (root / "reference_initial_survey/DONE").exists(),
              "reference_initial_survey": survey, "baseline_queries_recorded": len(baseline),
              "placement_query_counts": (exposures // 5).tolist(), "outcomes": outcomes,
              "restore_checks": read_jsonl(root / "restore_checks.jsonl"),
              "released_restore_checks": read_jsonl(root / "released_restore_checks.jsonl"), "repeat_controls": repeats,
              "sampler_parity": read_jsonl(root / "sampler_parity.jsonl"), "counterfactuals": interventions,
              "no_placement_snapshot": read_jsonl(root / "snapshot-missing.jsonl"),
              "scope": "4 development cases; instrumented routing is an auxiliary approximation; released follow-up and reference survey use untouched-public-server actions; no population inference"}
    (derived / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                             "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.titlesize": 13})
        fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), sharey=True, layout="constrained")
        for p, ax in enumerate(axes):
            values = counts[p] / max(int(exposures[p]), 1)
            im = ax.imshow(values, vmin=0, vmax=1, aspect="auto", cmap="viridis")
            ax.set_title(f"Placement {'true' if p else 'false'} ({exposures[p]//5} queries)")
            ax.set_xticks(range(4));ax.set_xlabel("Expert ID (not semantic skill)")
            ax.set_yticks(range(18))
            if p == 0:
                ax.set_ylabel("Layer (zero-based)")
        fig.colorbar(im, ax=axes.tolist(), label="Fraction of executed action-token positions")
        fig.suptitle("Frozen SkillNet: final-denoising routing, baseline only")
        fig.savefig(derived / "expert_utilization.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(["#3b528b", "#21918c", "#5ec962", "#fde725"])
        norm = BoundaryNorm(np.arange(5) - .5, 4)
        for case in sorted(root.glob("task*-init*")):
            queries = read_jsonl(case / "baseline/queries.jsonl")
            if not queries:
                continue
            routes = np.asarray([r["response"]["diagnostics"]["router_top1"] for r in queries])[:, :, 0].T
            truth = np.asarray([r["stage_truth"] for r in queries], dtype=int).T
            fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [4, 1]}, layout="constrained")
            end = queries[-1]["step"] + 5
            im = ax[0].imshow(routes, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm,
                              extent=[0, end, 17.5, -.5])
            ax[0].set_ylabel("Layer");ax[0].set_yticks([0, 5, 10, 17])
            ax[0].set_title(case.name + ": first executed token, final denoising iteration")
            fig.colorbar(im, ax=ax[0], ticks=range(4), label="Expert ID")
            ax[1].imshow(truth, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1,
                         extent=[0, end, 2.5, -.5])
            ax[1].set_yticks([0, 1, 2], labels=["Grasp proxy", "Placement", "Fixture goal"])
            ax[1].set_xlabel("Control step (5 steps per query)")
            fig.savefig(derived / f"{case.name}_routing_timeline.png", dpi=160)
            plt.close(fig)
        if survey:
            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
            names = ["wrong_skill_ids", "no_skills", "noise_only"]
            colors = ["#3b528b", "#21918c", "#d28e00"]
            positions = np.arange(len(survey))
            for offset, (name, color) in enumerate(zip(names, colors)):
                rows = [next(c for c in r["conditions"] if c["condition"] == name) for r in survey]
                x = positions + (offset - 1) * .24
                axes[0].bar(x, [r["action_rms_by_family"][0] for r in rows], width=.24, color=color, label=name)
                axes[1].bar(x, [r["instrumented_route_change_fraction"] for r in rows], width=.24, color=color)
            axes[0].set_yscale("log");axes[0].set_ylabel("Translation-command RMS\nuntouched public server")
            axes[0].legend(fontsize=9, ncol=3)
            axes[1].set_ylabel("Route-change fraction\ninstrumented auxiliary readout")
            axes[1].set_ylim(0, 1)
            axes[1].set_xticks(positions, labels=[f"Task {r['task_number']:02d}" for r in survey])
            fig.suptitle("Matched initial observations: skill input vs noise perturbation (one initial state/task)")
            fig.savefig(derived / "initial_skill_sensitivity.png", dpi=160)
            plt.close(fig)
        geom = root / "skill_embedding_geometry.json"
        if geom.exists():
            g = json.loads(geom.read_text())
            fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
            values = np.asarray(g["cosine_similarity"])
            im = ax.imshow(values, vmin=-1, vmax=1, cmap="coolwarm")
            labels = ["padding"] + [f"raw ID {i}" for i in range(5)]
            ax.set_xticks(range(6), labels=labels, rotation=40);ax.set_yticks(range(6), labels=labels)
            for i in range(6):
                for j in range(6):ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, label="Cosine similarity")
            ax.set_title("Released skill embedding table\nDistinct vectors do not establish learned skill semantics")
            fig.savefig(derived / "skill_embedding_cosine.png", dpi=160)
            plt.close(fig)
        import imageio.v2 as imageio
        for case in sorted(root.glob("task*-init*")):
            for branches, tag, source in [(("original", "remaining_instruction"), "comparison", "Instrumented cohort"),
                                          (("released_original", "released_remaining"), "released_comparison", "Untouched public behavior")]:
                if not all((case / b / "result.json").exists() for b in branches):
                    continue
                fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
                for row, branch in enumerate(branches):
                    metadata = json.loads((case / branch / "result.json").read_text())
                    reader = imageio.get_reader(str(case / branch / "video.mp4"))
                    length = reader.count_frames()
                    for col, frame in enumerate([0, length // 2, length - 1]):
                        axes[row, col].imshow(reader.get_data(frame)[:, :224])
                        label = "Full instruction" if row == 0 else "Remaining instruction"
                        axes[row, col].set_title(f"{label}: step {metadata['start_step']+5*frame}\nsuccess={metadata['success']}", fontsize=10)
                        axes[row, col].axis("off")
                    reader.close()
                fig.suptitle(f"{source}: {case.name}, identical reached-state intervention", fontsize=12)
                fig.savefig(derived / f"{case.name}_{tag}.png", dpi=150)
                plt.close(fig)
    except ImportError:
        result["figure_note"] = "matplotlib not installed; CSV remains available"
    lines = ["# SkillNet intervention collection", "", f"Main cohort complete: {result['collection_complete']}",
             f"Released follow-up complete: {result['released_followup_complete']}",
             f"Untouched-public-server initial survey complete: {result['reference_survey_complete']}",
             f"Baseline queries recorded: {len(baseline)}", "", "## Branch outcomes", "",
             "|Case|Branch|Success|Start|End|", "|---|---|---:|---:|---:|"]
    for row in outcomes:
        lines.append(f"|{row['case']}|{row['branch']}|{int(row['success'])}|{row['start_step']}|{row['end_step']}|")
    lines += ["", "## Interpretation limits", "", "Expert ID is not a named skill. The plot is descriptive and uses correlated queries, not independent episodes.",
              "Remaining-instruction branches use evaluation-only knowledge of what remains; this is an oracle diagnosis, not a deployable method.",
              "No causal branch claim is allowed without restore checks and the unchanged-instruction repeat control."]
    (derived / "report.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    r = analyze(args.root)
    print(json.dumps({k: r[k] for k in ("collection_complete", "released_followup_complete", "reference_survey_complete", "baseline_queries_recorded", "placement_query_counts", "repeat_controls")}))
