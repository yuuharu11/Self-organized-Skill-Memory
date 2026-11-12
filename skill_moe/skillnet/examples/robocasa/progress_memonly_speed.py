#!/usr/bin/env python3
"""Append completed worker counts for the RM-SPEED three-arm run."""
import datetime
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
progress = Path(sys.argv[2]) if len(sys.argv) > 2 else out / "progress.md"
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
lines = [f"\n## {stamp} UTC"]
for cond in ("x1p5", "x2p0", "nohist"):
    rows = {}
    for p in (out / cond / "eval/workers").glob("worker-*.json"):
        try:
            for row in json.loads(p.read_text()):
                if row.get("status") == "complete":
                    rows[(row["task"], int(row["trial"]))] = row
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    lines.append(f"\n### {cond}")
    for task in ("PickPlaceCounterToCabinet", "OpenDrawer", "CloseFridge"):
        subset = [r for (t, _), r in rows.items() if t == task]
        lines.append(f"- {task}: {sum(bool(r['success']) for r in subset)}/{len(subset)} success; {len(subset)}/12 complete")
progress.parent.mkdir(parents=True, exist_ok=True)
with progress.open("a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
