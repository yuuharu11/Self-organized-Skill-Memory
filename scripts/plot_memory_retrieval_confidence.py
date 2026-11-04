"""Plot saved retrieval confidence proxies without policy training or inference."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax


def describe(values):
    return {"n": len(values), "mean": float(np.mean(values)),
            **dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"),
                       map(float, np.quantile(values, (0, .1, .25, .5, .75, .9, 1)))))}


def histogram(ax, values, bins, color, label, fill=False):
    counts, edges = np.histogram(values, bins=bins)
    assert counts.sum() == len(values), "plot limits omit observations"
    share = 100 * counts / len(values)
    ax.stairs(share, edges, color=color, linewidth=2, label=label)
    if fill:
        ax.stairs(share, edges, color=color, fill=True, alpha=.13)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=200, facecolor="white")
    fig.savefig(output / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fair", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [args.fair / "results.json", args.fair / "per_row.npz", args.completion / "diagnostic_arrays.npz"]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    history = json.loads(sources[0].read_text())
    rows = np.load(sources[1], allow_pickle=False)
    cache = np.load(sources[2], allow_pickle=False)
    all_values, summaries = {}, []
    for seed in range(3):
        stem = f"B_s{seed}"
        run = next(r for r in history["runs"] if r["split"] == "B" and r["seed"] == seed)
        tau = run["sources"]["instruction"]["temperature"]
        values = {c: {kind: rows[f"{stem}|instruction_{c}|conf_{key}"]
                      for kind, key in (("similarity", "sim"), ("concentration", "ent"))}
                  for c in ("D1", "D3")}
        q, k = (cache[f"{stem}|{key}"].astype(np.float32) for key in ("query", "unit_keys"))
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        k /= np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-8)
        scores = q @ k.T
        weights = softmax(scores / tau, axis=1)
        sorted_scores = np.sort(scores, axis=1)
        margin = sorted_scores[:, -1] - sorted_scores[:, -2]
        reconstructed = 1 + np.sum(weights * np.log(np.maximum(weights, 1e-30)), axis=1) / np.log(len(k))
        delta = float(np.max(np.abs(reconstructed - values["D3"]["concentration"])))
        if delta > .001:
            raise ValueError("reconstructed concentration fails historical replay")
        values["margin"] = margin
        values["weight"] = weights.max(axis=1)
        all_values[seed] = values
        cal = run["sources"]["instruction"]["calibration"]["A"]["unit"]["sim"]
        summaries.append({"seed": seed, "unit_count": len(k), "temperature": tau,
            "D1": {key: describe(v) for key, v in values["D1"].items()},
            "D3": {key: describe(v) for key, v in values["D3"].items()},
            "D3_top1_top2_cosine_gap": describe(margin), "D3_top1_softmax_weight": describe(values["weight"]),
            "D3_weight_at_most_half_fraction": float(np.mean(values["weight"] <= .5)),
            "D3_gap_below_0p05_fraction": float(np.mean(margin < .05)),
            "lowest_validation_similarity_edge": cal["edges"][0],
            "D3_fraction_in_lowest_validation_similarity_bin": float(np.mean(values["D3"]["similarity"] < cal["edges"][0])),
            "concentration_replay_max_abs_difference": delta})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
                         "axes.labelsize": 11, "svg.fonttype": "none"})
    colors = {"D1": "#345B83", "D3": "#D37432"}
    labels = {"D1": "Known tasks: Object + Goal", "D3": "Held-out suite: Spatial"}
    def distributions(seeds, stem):
        fig, axes = plt.subplots(len(seeds), 2, figsize=(11, 3.5 * len(seeds)), squeeze=False)
        for row_index, seed in enumerate(seeds):
            for col, (key, title, xlabel) in enumerate((
                ("similarity", "Nearest-key similarity", "Maximum cosine similarity"),
                ("concentration", "Concentration across memory units", "1 - H(softmax scores) / log K"))):
                ax = axes[row_index, col]
                for cohort in ("D1", "D3"):
                    v = all_values[seed][cohort][key]
                    histogram(ax, v, np.linspace(0, 1, 31), colors[cohort],
                              f"{labels[cohort]} (median {np.median(v):.3f})", fill=True)
                    ax.axvline(np.median(v), color=colors[cohort], linestyle="--", linewidth=1.2)
                ax.set(xlim=(0,1), xlabel=xlabel, ylabel="Windows per bin (%)",
                       title=f"{title} | model seed {seed}")
                ax.legend(frameon=True, facecolor="white", framealpha=.9, edgecolor="none", fontsize=8.5)
                style(ax)
        fig.suptitle("Retrieval scores on known tasks and held-out Spatial", fontsize=16, y=.995)
        fig.text(.5, .015, "Each histogram is normalized separately. Scores are not calibrated correctness probabilities.\n"
                 "Overlapping action windows; D1 uses held-out episodes. Same models and train-only memory within each seed.",
                 ha="center", fontsize=9, color="#4A4A4A")
        fig.tight_layout(rect=(0,.09,1,.94))
        save(fig, args.output, stem)
    distributions([0], "confidence_distribution_seed0")
    distributions([0,1,2], "confidence_distribution_all_seeds")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    margin_max = max(float(v["margin"].max()) for v in all_values.values())
    for seed, color in enumerate(("#345B83", "#D37432", "#4B8A65")):
        for ax, key, bins in ((axes[0], "margin", np.linspace(0, margin_max + .001, 36)),
                              (axes[1], "weight", np.linspace(0, 1, 36))):
            v = all_values[seed][key]
            histogram(ax, v, bins, color, f"Seed {seed} (median {np.median(v):.3f})")
    for ax, xlabel, title in zip(axes, ("Top-1 cosine minus top-2 cosine", "Top-1 softmax weight"),
                                 ("Separation of the two nearest candidates", "Weight assigned to the nearest candidate")):
        ax.set(xlabel=xlabel, ylabel="Windows per bin (%)", title=title)
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    fig.suptitle("Candidate separation on held-out Spatial", fontsize=16)
    fig.text(.5,.015,"Reconstructed from archived float16 query/key vectors. Softmax weights are not correctness probabilities.\n"
             "A small gap can reflect multiple equally useful units; it does not establish retrieval failure.",ha="center",fontsize=9,color="#4A4A4A")
    fig.tight_layout(rect=(0,.1,1,.92))
    save(fig,args.output,"spatial_candidate_separation")
    payload = {"source_hashes": source_hashes, "runs": summaries,
               "similarity_and_concentration": "original saved scores; no reconstructed values substituted",
               "margin_and_weight": "reconstructed from archived float16 queries and unit keys; concentration replay checked",
               "scope": "descriptive overlapping-window distributions; not calibrated probabilities or independent trials"}
    (args.output/"summary.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps(summaries,indent=2))


if __name__ == "__main__":
    main()
