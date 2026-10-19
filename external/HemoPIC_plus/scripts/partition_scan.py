"""Does a finer ROI partition help? Grid over the cluster numbers, nested in the
outer folds of the Optuna search.

python scripts/partition_scan.py configs/full_aif.yaml --workers 20

For every outer fold f of tune_optuna.py --outer-folds 3 (seed 0), the base
configuration is the best Optuna trial of that fold (found without fold f).
Only (K_GM, K_WM), and optionally the tissue model, are varied over a grid, and
every grid point is run on the whole cohort. The grid point that is best on the
two other folds is evaluated on fold f, so the pooled result is held out and
directly comparable with the nested Optuna search and with oSVD.

Writes <out_root>/partition_scan/{per_patient.csv, summary.json} and prints a table.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import list_patients  # noqa: E402
from tune_optuna import MAP_NAMES, config_from_params, run_set  # noqa: E402


def outer_folds(pids, k=3):
    perm = list(np.random.default_rng(0).permutation(sorted(pids)))
    return {int(x): i % k for i, x in enumerate(perm)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--k-gm", type=int, nargs="*", default=[16, 32, 64])
    p.add_argument("--k-wm", type=int, nargs="*", default=[6, 12, 24])
    p.add_argument("--tissue", nargs="*", default=["tank"])
    p.add_argument("--patients", type=int, nargs="*")
    a = p.parse_args()
    base = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out = base.resolve("out_root")
    od = out / "partition_scan"
    od.mkdir(parents=True, exist_ok=True)
    pids = sorted(a.patients or list_patients(base.resolve("data_root")))
    fold = outer_folds(pids)

    f_pp = od / "per_patient.csv"
    done = {}
    if f_pp.exists():
        for r in csv.DictReader(open(f_pp, encoding="utf8")):
            done.setdefault(r["point"], {})[int(r["patient"])] = {m: float(r[f"AUC_{m}"]) for m in MAP_NAMES}
    bases = {}
    for k in sorted(set(fold.values())):
        params = json.loads((out / "optuna_cv" / f"fold{k}" / "test.json").read_text(encoding="utf8"))["tuning_best"]["params"]
        bases[k] = config_from_params(base, params)[0]
    grid = [(f, kg, kw, t) for f in bases for t in a.tissue for kg in a.k_gm for kw in a.k_wm]
    for f, b in bases.items():
        if (f, b.k_gm, b.k_wm, "tank") not in grid:
            grid.append((f, b.k_gm, b.k_wm, "tank"))
    ex = ProcessPoolExecutor(max_workers=a.workers)
    for f, kg, kw, t in grid:
        name = f"f{f}_gm{kg}_wm{kw}_{t}"
        if name in done and len(done[name]) == len(pids):
            continue
        t0 = time.time()
        cfg = replace(bases[f], k_gm=kg, k_wm=kw)
        d = od / "runs" / name
        res = run_set(ex, cfg, t, pids, d)
        shutil.rmtree(d, ignore_errors=True)
        done[name] = {pid: r["auc"] for pid, r in res.items()}
        with open(f_pp, "w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh)
            w.writerow(["point", "patient", *[f"AUC_{m}" for m in MAP_NAMES]])
            for pt, rows in done.items():
                for pid, auc in rows.items():
                    w.writerow([pt, pid, *[auc.get(m) for m in MAP_NAMES]])
        mean = {m: np.nanmean([v[m] for v in done[name].values()]) for m in MAP_NAMES}
        print(f"  {name:20s} n={len(res)}  CBF {mean['CBF']:.3f}  CBV {mean['CBV']:.3f}  MTT {mean['MTT']:.3f}  "
              f"mean {np.mean(list(mean.values())):.4f}  {time.time() - t0:.0f}s", flush=True)
    ex.shutdown()

    def score(point, ps):
        return float(np.nanmean([np.mean([done[point][p][m] for m in MAP_NAMES]) for p in ps if p in done[point]]))

    held, chosen = {}, {}
    for k in sorted(set(fold.values())):
        points = [f"f{f}_gm{kg}_wm{kw}_{t}" for f, kg, kw, t in grid if f == k and f"f{f}_gm{kg}_wm{kw}_{t}" in done]
        train = [p for p in pids if fold[p] != k]
        test = [p for p in pids if fold[p] == k]
        best = max(points, key=lambda pt: score(pt, train))
        chosen[k] = best
        for p in test:
            if p in done[best]:
                held[p] = done[best][p]
    osvd = {}
    f = out / "osvd" / "roc.csv"
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf8")):
            osvd.setdefault(int(r["patient"]), {})[r["map"]] = float(r["auc"])
    nested = {}
    f = out / "optuna_cv"
    for fd in sorted(f.glob("fold*")):
        for r in csv.DictReader(open(fd / "test_per_patient.csv", encoding="utf8")):
            if r["variant"].startswith("best1"):
                nested[int(r["patient"])] = {m: float(r[f"AUC_{m}"]) for m in MAP_NAMES}

    def mean_auc(d, ps):
        return {m: float(np.nanmean([d[p][m] for p in ps])) for m in MAP_NAMES} | \
               {"mean": float(np.nanmean([np.mean([d[p][m] for m in MAP_NAMES]) for p in ps]))}

    def paired(x, y, ps):
        r = {}
        for m in list(MAP_NAMES) + ["mean"]:
            d = np.array([(np.mean([x[p][q] for q in MAP_NAMES]) - np.mean([y[p][q] for q in MAP_NAMES])) if m == "mean"
                          else x[p][m] - y[p][m] for p in ps])
            d = d[np.isfinite(d)]
            r[m] = {"diff": float(d.mean()), "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
        return r

    common = sorted(set(held) & set(osvd) & set(nested))
    report = {"grid": {pt: mean_auc(done[pt], sorted(done[pt])) for pt in done}, "chosen_per_fold": chosen,
              "held_out_all": mean_auc(held, sorted(held)), "n_held": len(held),
              "on_osvd_patients": {"partition_scan": mean_auc(held, common), "nested_optuna": mean_auc(nested, common),
                                   "oSVD": mean_auc(osvd, common), "n": len(common)},
              "paired": {"scan - oSVD": paired(held, osvd, common), "scan - nested_optuna": paired(held, nested, common)}}
    (od / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print("grid (mean over all patients; descriptive, not held out):")
    for pt, s in report["grid"].items():
        print(f"  {pt:20s} CBF {s['CBF']:.3f}  CBV {s['CBV']:.3f}  MTT {s['MTT']:.3f}  mean {s['mean']:.4f}")
    print("chosen per outer fold:", chosen)
    for k, s in report["on_osvd_patients"].items():
        if k != "n":
            print(f"  held out, {report['on_osvd_patients']['n']} pts  {k:16s} CBF {s['CBF']:.3f}  CBV {s['CBV']:.3f}  MTT {s['MTT']:.3f}  mean {s['mean']:.4f}")
    for k, d in report["paired"].items():
        print(f"  {k:24s} " + "  ".join(f"{m} {v['diff']:+.3f} (p={v['p']:.2g})" for m, v in d.items()))


if __name__ == "__main__":
    main()
