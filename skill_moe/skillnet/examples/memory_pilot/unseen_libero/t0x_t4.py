#!/usr/bin/env python3
"""T4: evaluate Expert-displacement t0 maps using the existing KQ2-CL components."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import kq2cl_common as C
import kq2cl_closed as K
import kq2cl_expert as E
import expert_ex3 as EX
import t0x_closed as CL
from t0x_common import OUT, calibrate_m_radius, progress, write_json

T4 = OUT / "t4"
CAL_PATH = T4 / "calibration.json"
PROBE_T0 = .30
QUERY = "iii"
_REAL_POPEN = subprocess.Popen
BASE_CL_OUT = C.OUT


def t4progress(message: str, **fields):
    progress("t4", message, **fields)
    T4.mkdir(parents=True, exist_ok=True)
    row = {"jst": __import__("t0x_common").timestamp(), "message": message, **fields}
    with (T4 / "progress.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def calibrate(device: str = "cuda:0", batch: int = 1) -> dict:
    """Compute one deterministic Dev probe draw; never uses the Eval rows."""
    if CAL_PATH.exists():
        result = json.loads(CAL_PATH.read_text(encoding="utf-8"))
        t4progress("reuse existing Dev-only calibration", p_b=result["p_b"], m_b=result["m_b"])
        return result
    from pathlib import Path as _Path

    selected = C.read_json(C.OUT / "expert/select.json")["selected"]
    expert = EX.LiveExpert()
    expert.load(_Path(selected["checkpoint"]))
    rows, _, targets, _, _, _, drafts = E.split_inputs("dev2")
    draft = np.asarray(drafts[QUERY], np.float32)
    obs = expert.base.obs_batch(rows, np.arange(len(targets)))
    ids = np.arange(len(draft), dtype=np.int64) + 60_000_000
    noise = expert.exp3._seed_noise(ids)  # use the first fixed T2 noise seed only
    out = expert.bridge(obs, draft, noise, np.full(len(draft), PROBE_T0, np.float32), batch=batch)
    m = np.linalg.norm(out[..., :6] - draft[..., :6], axis=(-1, -2)).astype(np.float64)
    # This held-out-from-Expert Dev confidence prediction is from the fixed KQ2-CL iii head.
    prior = np.load(OUT / "t2" / "displacement-calibration.npz")
    ehat = np.asarray(prior["dev2_iii_ehat"], np.float64)
    if len(ehat) != len(m):
        raise AssertionError(f"Dev ehat/m count mismatch: {len(ehat)} vs {len(m)}")
    base = K.conditions()["A"]
    b = float(base["tpol"]["b"])
    if abs(b - .06849413737654686) > 1e-7:
        raise AssertionError(f"T4 expected the fixed K-x1 iii fallback b=.0684941, got {b}")
    result = calibrate_m_radius(ehat, m, b)
    result.update({"query": QUERY, "expert_checkpoint": selected["checkpoint"],
                   "expert_selection": selected, "probe_t0": PROBE_T0,
                   "noise_seed_rule": "T2 first fixed draw: exp3._seed_noise(arange(n)+60000000)",
                   "ehat_gt_b_count": int(np.sum(ehat > b)),
                   "m_quantiles_p05_p25_p50_p75_p95": np.quantile(m, [.05, .25, .5, .75, .95]).tolist(),
                   "ehat_quantiles_p05_p25_p50_p75_p95": np.quantile(ehat, [.05, .25, .5, .75, .95]).tolist(),
                   "gpu_device": device, "batch": int(batch)})
    write_json(CAL_PATH, result)
    np.savez_compressed(T4 / "calibration_arrays.npz", m=m.astype(np.float32), ehat=ehat.astype(np.float32))
    t4progress("Dev-only calibration complete", dev_rows=len(m), p_b=result["p_b"], m_b=result["m_b"], b=b)
    return result


def build_specs(calibration: dict | None = None) -> dict:
    if calibration is None:
        calibration = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    base = K.conditions()["A"]  # same iii K-x1, K1:500, bundle_iii, decoder and b as T1 A-add
    b = float(base["tpol"]["b"])
    if abs(b - float(calibration["b"])) > 1e-10:
        raise AssertionError("T4 calibration / serving fallback threshold mismatch")
    out = {}
    for name, formula in (("Aexp", "m_radius"), ("Amax", "max_radius")):
        out[name] = base | {"tpol": {"formula": formula, "b": b, "calibration": str(CAL_PATH)}}
    return out


def _popen_with_t0x_server(args, *pos, **kw):
    if isinstance(args, (list, tuple)):
        args = list(args)
        for i, value in enumerate(args):
            if str(value).endswith("/kq2cl_serve.py") or str(value) == str(C.HERE / "kq2cl_serve.py"):
                args[i] = str(C.HERE / "t0x_serve.py")
                break
    return _REAL_POPEN(args, *pos, **kw)


def run_suite(suite: str, conditions: list[str], *, smoke: bool, workers: int = 6,
              fraction: float = .20, cpus: str = "24-31,32-39", retries: int = 3):
    """Run/resume T4 trials; missing or error trials are retried without touching prior stages."""
    if suite not in ("libero90", "libero10"):
        raise ValueError(suite)
    C.OUT = BASE_CL_OUT
    stage_out = T4 / ("smoke" if smoke else "closed-loop")
    stage_out.mkdir(parents=True, exist_ok=True)
    spec = build_specs()
    spec_path = stage_out / f"conditions-{suite}.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise RuntimeError(f"refusing to overwrite a different T4 condition spec: {spec_path}")
    else:
        write_json(spec_path, spec)
    C.OUT = stage_out if smoke else T4
    for name in ("Aexp", "Amax"):
        K.PLAN[suite][name] = range(10)
    old_popen = K.subprocess.Popen
    K.subprocess.Popen = _popen_with_t0x_server
    prior_threads = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    try:
        for attempt in range(1, retries + 1):
            t4progress("closed-loop suite starting", suite=suite, conditions=conditions, smoke=smoke,
                       attempt=attempt, workers_per_condition=(1 if smoke else workers),
                       fraction=fraction, gpu=K.gpu_used())
            # Start one condition at a time. Forking two ProcessPoolExecutors concurrently from
            # K.run's condition threads can leave an idle worker/server alive after all JSONL rows
            # have been written. Per-condition sequential execution keeps each worker pool joined
            # before the next server starts and is still restartable from the same trial files.
            cpu_sets = cpus.split(",")
            for index, name in enumerate(conditions):
                K.run([name], workers=(1 if smoke else workers), fraction=fraction, smoke=smoke,
                      cpus=[cpu_sets[index % len(cpu_sets)]], spec_path=str(spec_path), suite=suite)
            base = C.OUT / (("closed-loop-smoke" if suite == "libero90" else "closed-loop-smoke-libero10")
                            if smoke else f"closed-loop/{suite}")
            missing = {}
            for name in conditions:
                records = K.read_trials(base / name)
                n_expected = 2 if smoke else 100
                complete = sum(r.get("status") == "complete" for r in records.values())
                if complete < n_expected:
                    missing[name] = {"complete": complete, "expected": n_expected}
            if not missing:
                t4progress("closed-loop suite complete", suite=suite, conditions=conditions,
                           smoke=smoke, attempt=attempt)
                return
            t4progress("retrying incomplete or failed trials", suite=suite, attempt=attempt, missing=missing,
                       oom_log_matches=_count_oom_logs(base / "servers"))
        raise RuntimeError(f"T4 retries exhausted for {suite}: {missing}")
    finally:
        K.subprocess.Popen = old_popen
        for k, v in prior_threads.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _count_oom_logs(root: Path) -> int:
    n = 0
    for path in root.glob("*.log") if root.exists() else []:
        try:
            text = path.read_text(errors="replace").lower()
        except OSError:
            continue
        n += text.count("out of memory") + text.count("resource_exhausted") + text.count("cuda_error_out_of_memory")
    return n


def run_all(*, workers=6, fraction=.20, cpus="24-31,32-39", batch=1):
    if not CAL_PATH.exists():
        calibrate(batch=batch)
    # Smoke each t0 mapping before its resumable 100-trial evaluation.
    run_suite("libero90", ["Aexp", "Amax"], smoke=True, workers=1, fraction=fraction, cpus=cpus)
    for name in ("Aexp", "Amax"):
        path = T4 / "smoke" / "closed-loop-smoke" / name / "trials.jsonl"
        rows = CL.read_trials(path)
        if len([r for r in rows if r.get("status") == "complete"]) != 2:
            raise AssertionError(f"T4 smoke did not complete 2 trials for {name}: {path}")
    run_suite("libero90", ["Aexp", "Amax"], smoke=False, workers=workers, fraction=fraction, cpus=cpus)
    run_suite("libero10", ["Aexp", "Amax"], smoke=False, workers=workers, fraction=fraction, cpus=cpus)
    # Mark completion in the aggregate only after both suites passed their exact-count checks.
    import t0x_report as R
    aggregate = R.build()
    aggregate.setdefault("status", {})["t4_done_marker"] = True
    write_json(OUT / "results.json", aggregate)
    (OUT / "report.md").write_text(R.md_report(aggregate), encoding="utf-8")
    write_json(T4 / "results.json", aggregate.get("t4", {}))
    (T4 / "DONE").write_text(f"DONE {__import__('t0x_common').timestamp()} JST\n", encoding="utf-8")
    t4progress("DONE", suites={"libero90": 200, "libero10": 200}, calibration=CAL_PATH.name,
               oom_log_matches={s: _count_oom_logs(T4 / "closed-loop" / s / "servers")
                                for s in ("libero90", "libero10")})
    return aggregate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("calibrate", "smoke", "run", "all"))
    ap.add_argument("--suite", choices=("libero90", "libero10"), default="libero90")
    ap.add_argument("--conditions", default="Aexp,Amax")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fraction", type=float, default=.20)
    ap.add_argument("--cpus", default="24-31,32-39")
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args()
    os.environ.setdefault("SKILLNET_SUBTASK_GENERATE", "0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if a.command == "calibrate":
        print(json.dumps(calibrate(batch=a.batch), indent=2))
    elif a.command == "smoke":
        if not CAL_PATH.exists():
            calibrate(batch=a.batch)
        run_suite(a.suite, a.conditions.split(","), smoke=True, workers=1,
                  fraction=a.fraction, cpus=a.cpus)
    elif a.command == "run":
        if not CAL_PATH.exists():
            calibrate(batch=a.batch)
        run_suite(a.suite, a.conditions.split(","), smoke=False, workers=a.workers,
                  fraction=a.fraction, cpus=a.cpus)
    else:
        run_all(workers=a.workers, fraction=a.fraction, cpus=a.cpus, batch=a.batch)


if __name__ == "__main__":
    main()
