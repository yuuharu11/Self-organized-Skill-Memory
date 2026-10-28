"""Write LIBERO's path config before its first import (which otherwise prompts)."""

import importlib.util
import os
from pathlib import Path

import yaml


def main():
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("LIBERO is not installed")
    roots = [Path(p) / "libero" for p in spec.submodule_search_locations]
    root = next((p for p in roots if (p / "assets").is_dir()), None)
    if root is None:
        raise SystemExit("LIBERO simulator assets are missing; an editable full LIBERO checkout is required")
    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "assets": str(root / "assets"),
        "datasets": str(Path(os.environ["HF_LEROBOT_HOME"]).parent / "libero" / "datasets"),
    }
    Path(config["datasets"]).mkdir(parents=True, exist_ok=True)
    # Regenerate paths when the active LIBERO installation changes.
    (config_dir / "config.yaml").write_text(yaml.safe_dump(config))


if __name__ == "__main__":
    main()
