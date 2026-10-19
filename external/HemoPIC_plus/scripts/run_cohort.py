"""Fit every subject of the cohort with one experiment config.

python scripts/run_cohort.py configs/baseline.yaml
python scripts/run_cohort.py configs/full.yaml --patients 1 22 --workers 4
python scripts/run_cohort.py configs/full.yaml --data-root D:/ISLES2017_Training

Outputs go to <out_root>/<config name>/training_XX/ (HemoPIC layout).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import list_patients  # noqa: E402
from hemopic_plus.pipeline import run_subject  # noqa: E402


def _work(args):
    cfg, pid, out_root = args
    t0 = time.time()
    try:
        res = run_subject(cfg, pid, out_root)
        res["seconds"] = round(time.time() - t0, 1)
        return res
    except Exception as e:
        return {"patient": pid, "error": repr(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--workers", type=int)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    a = p.parse_args()

    overrides = {k: v for k, v in (("workers", a.workers), ("data_root", a.data_root), ("out_root", a.out_root)) if v is not None}
    if a.patients:
        overrides["patients"] = a.patients
    cfg = load_config(a.config, overrides)

    data_root = cfg.resolve("data_root")
    if not data_root.exists():
        raise SystemExit(f"data_root not found: {data_root} (set data_root in the config or pass --data-root)")
    out_root = cfg.resolve("out_root") / cfg.name
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf8")

    pids = cfg.patients or list_patients(data_root)
    workers = cfg.workers or os.cpu_count() or 1
    print(f"[{cfg.name}] {len(pids)} subjects, {workers} workers -> {out_root}")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_work, (cfg, pid, out_root)) for pid in pids]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if "error" in r:
                print(f"  FAIL {r['patient']}: {r['error']}")
            else:
                print(f"  ok {r['patient']:>2}  dt={r['dt']:.3f}s  rois={r['n_rois']}  failed={r['n_failed']}  "
                      f"median_loss={r['median_loss']:.4f}  {r['seconds']}s")
    results.sort(key=lambda r: r["patient"])
    (out_root / "run_log.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    n_err = sum("error" in r for r in results)
    print(f"[{cfg.name}] done in {time.time() - t0:.0f}s, {len(results) - n_err} ok, {n_err} failed")


if __name__ == "__main__":
    main()
