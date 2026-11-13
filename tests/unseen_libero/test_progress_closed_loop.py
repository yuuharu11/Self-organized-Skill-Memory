"""Progress closed-loop runner: concurrent progress writes must not race.

Four rollout lanes share one progress.json. The writer must use a unique
temporary file per call; a shared ".tmp" path lets one lane rename it
before another reaches replace(), aborting the run with FileNotFoundError.
"""
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skill_moe/skillnet/examples/memory_pilot/unseen_libero/progress_closed_loop.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("progress_closed_loop_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_concurrent_progress_writes_all_land(tmp_path):
    target = tmp_path / "cond" / "progress.json"

    def worker(i):
        mod.write(target, {"lane": i, "updated_unix": float(i)})

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, range(32)))
    assert target.is_file()
    import json

    json.loads(target.read_text())
