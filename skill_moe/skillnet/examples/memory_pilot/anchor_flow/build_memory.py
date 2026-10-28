#!/usr/bin/env python3
"""Fit the frozen S2 M1 codec and seed its M6 memory from LIBERO-90 only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1]
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from memory import fit_memory, save_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero90-npz", required=True, help="Normalized LIBERO-90-only M1/M6 fitting rows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    source = Path(args.libero90_npz)
    with np.load(source, allow_pickle=True) as archive:
        data = {name: archive[name] for name in archive.files}
    if "suite" in data and not np.all(np.asarray(data["suite"]).astype(str) == "libero90"):
        raise ValueError("input contains non-LIBERO-90 rows")
    memory = fit_memory(data, seed=args.seed, epochs=args.epochs)
    save_memory(memory, args.output)
    print({"output": str(Path(args.output).resolve()), "rows": len(data["actions"]),
           "units": len(memory._centers), "memory_items": len(memory.memory),
           "epsilon": memory.epsilon, "backbone_sha256": memory and __import__("memory").backbone_digest(memory)})


if __name__ == "__main__":
    main()
