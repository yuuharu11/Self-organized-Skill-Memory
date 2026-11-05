"""Complete compact retrieval metrics from saved row IDs and predictions."""
import json
import numpy as np
from pathlib import Path
import run


def _clean_nonfinite(value):
    if isinstance(value, dict):
        return {k: _clean_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_nonfinite(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    out = run.OUT / "production"
    rows = run._load_rows()
    cand = np.load(out / "candidate_rows.npz", allow_pickle=False)
    pred = np.load(out / "predictions.npz", allow_pickle=False)
    gen = np.load(out / "generation_predictions.npz", allow_pickle=False)
    qids = np.asarray(cand["query_rows"], np.int64)
    if not np.array_equal(qids, pred["query_rows"]) or not np.array_equal(qids, gen["query_rows"]):
        raise AssertionError("saved per-row query ordering differs")
    truth = np.asarray(pred["truth"], np.float32)
    pool = np.asarray(np.load(run.MEMORY / "units.npz", allow_pickle=False)["pool_row_ids"], np.int64)
    all_mse = run._action_mse_rows(rows, qids, pool)
    all_mse[np.asarray(rows["task"][qids])[:, None] == np.asarray(rows["task"][pool])[None]] = np.inf
    methods = {}
    for name in (k[:-5] for k in cand.files if k.endswith("_rows") and f"{k[:-5]}_scores" in cand.files):
        cr = np.asarray(cand[f"{name}_rows"], np.int64)
        sc = np.asarray(cand[f"{name}_scores"], np.float32)
        cfkey = f"{name}_confidence"
        conf = np.asarray(pred[cfkey], np.float32) if cfkey in pred.files else sc[:, 0]
        methods[name] = run._best_metrics(name, qids, cr, sc, rows, all_mse, conf=conf)
    for name in ("M1_text_reference",):
        if name not in methods and f"{name}_rows" in cand.files:
            cr=np.asarray(cand[f"{name}_rows"],np.int64);sc=np.asarray(cand[f"{name}_scores"],np.float32)
            methods[name]=run._best_metrics(name,qids,cr,sc,rows,all_mse,conf=sc[:,0])
    zindex = {str(n): i for i, n in enumerate(gen["method_names"].astype(str))}
    for name in ("direct_regression", "zero_draft", "normal_pi05"):
        if name == "direct_regression":
            action = np.asarray(gen["drafts"][:, zindex[name], :, :7], np.float32)
        elif name == "zero_draft":
            action = np.zeros_like(truth)
        else:
            action = np.asarray(gen["normal"], np.float32)
        methods[name] = run._prediction_only(action, truth, name)
    base = run._baseline_predictions(rows, qids, pool, run.TOPK / "predictions_and_errors.npz")
    methods["hold_last"] = run._prediction_only(base["hold_last"], truth, "hold_last")
    methods["previous_chunk_copy"] = run._prediction_only(base["previous_chunk_copy"], truth, "previous_chunk_copy")
    hard=np.asarray(pred["hard_mask"],bool);episodes=run._episode_ids(rows,qids)
    metrics=run._metrics_summary(rows,qids,hard,all_mse,methods,episodes)
    conf_metrics={}
    for name,m in methods.items():
        if m.get("confidence") is not None:
            conf_metrics[name]=run._risk_metrics(m["confidence"],m["top1_mse"],episodes)
    summary=json.loads((out/"summary.json").read_text())
    summary["metrics"]=metrics
    summary["confidence_by_method"]=conf_metrics
    summary["metric_reconstruction"]="Recomputed from saved candidate row IDs and per-row truth; candidate indexing uses dataset row IDs. This repairs the initial summary writer omission; raw predictions/candidates are unchanged."
    (out/"summary.json").write_text(json.dumps(_clean_nonfinite(summary),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"metrics":metrics,"confidence_by_method":conf_metrics},ensure_ascii=False))


if __name__ == "__main__":
    import torch
    torch.set_num_threads(16)
    main()
