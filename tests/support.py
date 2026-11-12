"""Shared helpers: repository paths and loading example scripts that are not importable packages."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "skill_moe/skillnet/examples"
MEMORY_PILOT = EXAMPLES / "memory_pilot"
PI05 = MEMORY_PILOT / "pi05_memory"
UNSEEN = MEMORY_PILOT / "unseen_libero"


def add_path(path: Path | str) -> None:
    """Put a script directory on sys.path so its sibling imports resolve."""
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def load_script(path: Path | str):
    """Load a script file as a module once, registered in sys.modules for dataclasses and pickling."""
    path = Path(path).resolve()
    name = "_script_" + "_".join(path.relative_to(REPO).with_suffix("").parts)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
