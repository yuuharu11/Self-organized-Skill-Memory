"""Command line entry point for the motion-primitive quick experiment."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .data import load_recording, make_synthetic_dataset
from .experiment import run_grid, unique_output_dir, write_report


def find_latest_valid_record(root: str | Path = "/work/.skillnet-state/outputs/memory-pilot") -> Path | None:
    """Find newest record run whose own report marks it valid, excluding invalid snapshots."""
    candidates = []
    for run_dir in Path(root).glob("record-*"):
        report = run_dir / "report.md"
        if not report.exists():
            continue
        content = report.read_text(encoding="utf-8")
        if "無効" in content:
            continue
        if re.search(r"(?:判定\s*[:：].*有効|^#.*有効)", content, flags=re.MULTILINE):
            candidates.append(run_dir)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", type=Path, help="valid recorded rollout run")
    source.add_argument("--synthetic", action="store_true", help="run on generated smoke data")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beta-kl", type=float, nargs="+", default=[1e-3],
                        help="one or more beta values; multiple values form a KL-weight sweep")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[8],
                        help="one or more z dimensions to compare")
    parser.add_argument("--mi-permutations", type=int, default=100,
                        help="label permutations for finite-sample MI null baselines")
    parser.add_argument("--cross-random-draws", type=int, default=8,
                        help="uniform training-donor draws per query for cross reconstruction")
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", help="cpu or an explicitly available CUDA device")
    parser.add_argument("--output-root", type=Path, default=Path("/work/.skillnet-state/outputs/memory-pilot"))
    args = parser.parse_args()
    if args.synthetic:
        dataset = make_synthetic_dataset(horizon=args.horizon)
        source_label = "synthetic"
    else:
        run_dir = args.run_dir or find_latest_valid_record(args.output_root)
        if run_dir is None:
            parser.error("no record-* report marked 有効; pass --run-dir after a valid record is available")
        dataset = load_recording(run_dir, horizon=args.horizon)
        source_label = str(run_dir)
    rows = run_grid(dataset, epochs=args.epochs, batch_size=args.batch_size,
                    eval_fraction=args.eval_fraction, device_name=args.device,
                    beta_kl_values=tuple(args.beta_kl), latent_dims=tuple(args.latent_dims),
                    mi_permutations=args.mi_permutations, cross_random_draws=args.cross_random_draws)
    output_dir = unique_output_dir(args.output_root)
    write_report(rows, output_dir, source=source_label)
    print(output_dir)


if __name__ == "__main__":
    main()
