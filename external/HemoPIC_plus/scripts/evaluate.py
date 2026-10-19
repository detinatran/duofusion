"""Evaluate one experiment (tracer nRMSE + lesion ROC) and write a summary.

python scripts/evaluate.py configs/full.yaml
python scripts/evaluate.py configs/full.yaml --data-root D:/ISLES2017_Training

Reads <out_root>/<name>/, writes <out_root>/<name>/eval/{per_patient.csv,summary.json}.
For the paper's remaining figures (CVT, tau, summary stats) the original
HemoPIC Eval/ scripts can be pointed at <out_root>/<name> unchanged.
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import MAPS, REGIONS, evaluate_subject, roc_auc  # noqa: E402
from hemopic_plus.io import list_patients  # noqa: E402


def _work(args):
    data_root, fit_root, pid = args
    try:
        return evaluate_subject(data_root, fit_root, pid)
    except Exception as e:
        print(f"  skip {pid}: {e!r}")
        return [], {}


def summarize(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)), "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int)
    a = p.parse_args()
    overrides = {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root), ("workers", a.workers)) if v is not None}
    cfg = load_config(a.config, overrides)
    data_root = cfg.resolve("data_root")
    fit_root = cfg.resolve("out_root") / cfg.name
    pids = [pid for pid in list_patients(fit_root) if (fit_root / f"training_{pid}" / "fit_curves.npz").exists()]
    if not pids:
        raise SystemExit(f"No fitted subjects in {fit_root}; run scripts/run_cohort.py first")

    rows, pooled = [], {}
    with ProcessPoolExecutor(max_workers=cfg.workers or os.cpu_count() or 1) as ex:
        for r, pl in ex.map(_work, [(data_root, fit_root, pid) for pid in pids]):
            rows.extend(r)
            for k, (y, s) in pl.items():
                pooled.setdefault(k, ([], []))
                pooled[k][0].append(y)
                pooled[k][1].append(s)

    out = fit_root / "eval"
    out.mkdir(exist_ok=True)
    with open(out / "per_patient.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "kind", "region", "method", "value", "nmse"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["patient"], r["kind"], r["region"], r["method"])))

    summary = {"name": cfg.name, "n_patients": len(pids), "tracer_nrmse": {}, "roc_auc_patient_mean": {}, "roc_auc_pooled": {}}
    for reg in REGIONS:
        summary["tracer_nrmse"][reg] = summarize([r["value"] for r in rows if r["kind"] == "tracer" and r["region"] == reg])
    for name, _ in MAPS:
        for method in ("model", "deconv"):
            summary["roc_auc_patient_mean"][f"{method}_{name}"] = summarize(
                [r["value"] for r in rows if r["kind"] == "roc" and r["region"] == name and r["method"] == method])
            if (method, name) in pooled:
                y = np.concatenate(pooled[(method, name)][0])
                s = np.concatenate(pooled[(method, name)][1])
                summary["roc_auc_pooled"][f"{method}_{name}"] = roc_auc(y, s)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")

    print(f"[{cfg.name}] {len(pids)} subjects")
    print("  tracer nRMSE mean: " + "  ".join(f"{k}={v.get('mean', float('nan')):.3f}" for k, v in summary["tracer_nrmse"].items()))
    for name, _ in MAPS:
        m, d = summary["roc_auc_patient_mean"][f"model_{name}"], summary["roc_auc_patient_mean"][f"deconv_{name}"]
        print(f"  ROC {name}: model {m.get('mean', float('nan')):.4f} (pooled {summary['roc_auc_pooled'].get('model_' + name, float('nan')):.4f})"
              f"   deconv {d.get('mean', float('nan')):.4f}")
    print("  ->", out / "summary.json")


if __name__ == "__main__":
    main()
