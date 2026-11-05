"""Render the selected saved model's attention maps without rerunning training."""
from pathlib import Path
import json
import torch

import run
from models import M1AttentionKey, M2SalientTokens


def main():
    out = run.OUT / "production"
    summary = json.loads((out / "summary.json").read_text())
    saved = torch.load(out / "models.pt", map_location="cpu", weights_only=False)
    selected = summary["decision"]["selected_model"]
    visual_name = selected if selected in ("M1_slots4", "M1_slots8", "M2_salient_maxsim") else saved["selected_m1_for_m3"]
    if visual_name == "M2_salient_maxsim":
        model = M2SalientTokens(dim=96, token_count=512)
    else:
        slots = int(visual_name.rsplit("slots", 1)[-1])
        model = M1AttentionKey(slots=slots, dim=96, token_count=512)
    model.load_state_dict(saved["models"][visual_name])
    model.to(run.DEVICE).eval()
    rows = run._load_rows()
    pred = __import__("numpy").load(out / "predictions.npz", allow_pickle=False)
    tokens = __import__("numpy").load(run.RUN / "tokens.npy", mmap_mode="r", allow_pickle=False)
    groups = __import__("numpy").load(run.RUN / "groups.npy", mmap_mode="r", allow_pickle=False)
    context = run._history(rows)
    examples = run._visualize_attention(out, rows, model, pred["query_rows"], pred["hard_mask"], context, tokens, groups)
    summary["attention_examples"] = {"model": visual_name, **examples}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"attention_model": visual_name, "examples": examples}, ensure_ascii=False))


if __name__ == "__main__":
    if run.DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(.30)
    torch.set_num_threads(16)
    main()
