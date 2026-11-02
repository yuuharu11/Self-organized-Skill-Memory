"""Fit frozen-feature achievement heads and paired intervention-effect heads."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from effect_controller import predict_probabilities


def transform_for(x, feature_dim=2048):
    # Preserve low-dimensional proprioception, causal deltas and remaining budget.
    # PCA is fitted only to the training partition.
    prefix = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=min(16, len(x)-1), random_state=7))])
    columns = ColumnTransformer([("prefix", prefix, slice(0, feature_dim)),
                                 ("causal_state", StandardScaler(), slice(feature_dim, x.shape[1]))])
    return Pipeline([("columns", columns), ("scale", StandardScaler())]).fit(x)


def fit_monitor(root):
    parents = []
    validation = []
    for result in sorted((root / "parents").glob("*/result.json")):
        row = json.loads(result.read_text())
        job = row["job"]
        if job["init"] in (20, 21):
            raise ValueError("test parent leakage")
        data = np.load(result.parent / "features.npz")
        (validation if job["split"] == "validation" else parents).append((job, data["vectors"], data["labels"]))
    if not parents:
        raise ValueError("no completed training parents")
    x = np.concatenate([row[1] for row in parents])
    y = np.concatenate([row[2] for row in parents]).astype(int)
    transform = transform_for(x, x.shape[1]-24)
    z = transform.transform(x)
    heads = []
    for k in range(3):
        heads.append(float(y[0,k]) if len(np.unique(y[:,k])) == 1 else
                     LogisticRegression(C=1., max_iter=1000, random_state=7).fit(z, y[:,k]))
    model = dict(transform=transform, heads=heads,
                 train_parents=[row[0]["directory"] for row in parents], target="verified ordered goals 0/1 and first-goal grasp")
    (root / "models").mkdir(exist_ok=True)
    (root / "models/monitor.pkl").write_bytes(pickle.dumps(model))
    report = dict(train_parents=model["train_parents"], train_observations=len(x), validation=[])
    for job, vx, vy in validation:
        p = np.asarray([predict_probabilities(model, vector) for vector in vx])
        report["validation"].append(dict(parent=job["directory"], n=len(vx),
            brier=[float(brier_score_loss(vy[:, k], p[:, k])) for k in range(3)],
            accuracy=[float(np.mean((p[:, k] >= .5) == vy[:, k])) for k in range(3)],
            positives=vy.sum(axis=0).astype(int).tolist(), predicted_positives=(p >= .8).sum(axis=0).tolist()))
    (root / "models/monitor_report.json").write_text(json.dumps(report, indent=2))
    return report


def fit_effect(root):
    monitor = pickle.loads((root / "models/monitor.pkl").read_bytes())
    grouped = {}
    for result in (root / "pairs").glob("*/result.json"):
        row = json.loads(result.read_text())
        job = row["job"]
        key = (job["parent"], job["snapshot"], job["seed"])
        grouped.setdefault(key, {})[job["arm"]] = row
    snapshots = {}
    for (parent, snapshot, seed), arms in grouped.items():
        if set(arms) != {"keep", "update"}:
            continue
        target = float(arms["update"]["success"]) - float(arms["keep"]["success"])
        snapshots.setdefault((parent, snapshot), []).append((seed, target, arms["keep"]["job"]["split"]))
    rows = []
    for (parent, snapshot), values in sorted(snapshots.items()):
        if {value[0] for value in values} != {17, 29}:
            continue
        vector = np.load(root / snapshot)["vector"]
        p = predict_probabilities(monitor, vector)
        rows.append(dict(parent=parent, snapshot=snapshot, split=values[0][2],
                         x=np.concatenate([vector, p]), y=float(np.mean([v[1] for v in values]))))
    train = [row for row in rows if row["split"] == "train"]
    if len(train) < 4 or len({row["parent"] for row in train}) < 2:
        raise ValueError("insufficient complete paired training groups")
    x = np.asarray([row["x"] for row in train])
    y = np.asarray([row["y"] for row in train])
    transform = transform_for(x, x.shape[1]-27)
    z = transform.transform(x)
    parent_ids = sorted({row["parent"] for row in train})
    rng = np.random.default_rng(7)
    heads = []
    for _ in range(12):
        draw = rng.choice(parent_ids, len(parent_ids), replace=True)
        indices = [i for parent in draw for i, row in enumerate(train) if row["parent"] == parent]
        heads.append(Ridge(alpha=10.).fit(z[indices], y[indices]))
    model = dict(transform=transform, heads=heads, train_parents=parent_ids,
                 target="mean paired difference in original whole-task success over common future seeds17,29")
    (root / "models/effect.pkl").write_bytes(pickle.dumps(model))
    report = dict(train_parents=parent_ids, train_snapshots=len(train),
        complete_snapshots=len(rows), target_counts={str(v):int(np.sum(y==v)) for v in np.unique(y)},
        pairs=[{k:v for k,v in row.items() if k!="x"} for row in rows])
    (root / "models/effect_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["monitor", "effect"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps((fit_monitor if args.phase == "monitor" else fit_effect)(args.root), indent=2))
