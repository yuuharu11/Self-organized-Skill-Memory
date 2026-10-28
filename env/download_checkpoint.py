"""Download one released checkpoint and record the resolved Hub revision."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("40", "90"))
    parser.add_argument("--revision", default="main")
    state_dir = Path(os.environ.get("SKILLNET_STATE_DIR", Path(__file__).resolve().parents[1] / ".skillnet-state"))
    parser.add_argument("--checkpoint-root", type=Path, default=state_dir / "checkpoints")
    args = parser.parse_args()
    repo = f"jsw19/SkillNet-LIBERO-{args.suite}"
    suffix = {
        "40": "pi05_libero_moe_skill_4_40/moe_add_balance_4_128_30000/29999",
        "90": "pi05_libero_moe_skill_4_90/moe_balance_4_32_90_20000/19999",
    }[args.suite]
    dest = args.checkpoint_root / suffix
    revision = HfApi().model_info(repo, revision=args.revision).sha
    snapshot_download(repo, revision=revision, local_dir=dest)
    (dest / "download_revision.json").write_text(json.dumps({"repo_id": repo, "revision": revision}, indent=2) + "\n")
    if not (dest / "params").is_dir() or not (dest / "assets").is_dir():
        raise SystemExit(f"Incomplete checkpoint: {dest}")
    print(dest.resolve())


if __name__ == "__main__":
    main()
