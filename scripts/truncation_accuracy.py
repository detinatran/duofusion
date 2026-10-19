"""Accuracy and stability under truncation, with cross-fitted thresholds.

python scripts/truncation_accuracy.py <results_dir> --yy 24 --lengths 25 30 35 40 45 [--out file.json]

Methods (volume grids in <results_dir>): duofusion<yy>[_T L], osvd<yy>_*, ssvd<yy>_*,
gamma_osvd<yy>_*, hybrid<yy>_* (scripts/hybrid_grids.py). References, independent of the
methods and computed on the full scan:
  core   yy=24: vendor core (rCBF < 0.3, mv24/); yy=18: the DWI core acquired right after
         the CTP (mv18/, truth_core_ml)
  hypo   vendor Tmax > 6 s volume (duofusion<yy>/per_patient.json)
  DEFUSE 3 decision (yy=24 only): the vendor decision on the full scan
Thresholds (rCBF tau, relative-Tmax d) are chosen per method on the FULL scans of the
training folds (5 folds over patients; median |log1p(volume) - log1p(reference)|) and
applied to the test fold at every scan length. Per length: median |log1p error| and
median |error| (ml) of core and hypo against the references, median |change| against the
method's own full scan, DEFUSE 3 flips against the own full scan and agreement with the
vendor decision.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METHODS = ("duofusion", "osvd", "ssvd", "gamma_osvd", "gamma_ssvd", "learned_ssvd", "hybrid")


def grids(res, method, yy, suf):
    d = res / ((f"duofusion{yy}" if suf == "full" else f"duofusion{yy}_{suf}") if method == "duofusion"
               else f"{method}{yy}_{suf}")
    out = {}
    for f in d.glob("training_*.npz"):
        z = np.load(f)
        g = {k: z[k][0] if (method == "duofusion" and k in ("core_grid", "hypo_grid", "union_grid")) else z[k]
             for k in ("core_grid", "hypo_grid", "union_grid")}
        out[int(f.stem.split("_")[1])] = g
    return out


def defuse3(core, union):
    return bool(core < 70 and (union / core >= 1.8 if core > 0 else True) and union - core >= 15)


def folds(groups, k=5, seed=0):
    keys = sorted(set(groups.values()), key=str)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    f = {g: i % k for i, g in enumerate(keys)}
    return {n: f[g] for n, g in groups.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--yy", default="24")
    p.add_argument("--lengths", type=int, nargs="+", default=[25, 30, 35, 40, 45])
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    R, yy = a.results, a.yy
    mvdir = R / ("mv24" if yy == "24" else "mv18")
    if not mvdir.exists():
        mvdir = R / f"duofusion_mv{yy}"
    mv = {int(r["patient"]): r for r in csv.DictReader(open(mvdir / "multiverse_per_patient.csv", encoding="utf8"))}
    dfj = {r["patient"]: r for r in json.load(open(R / f"duofusion{yy}" / "per_patient.json", encoding="utf8"))}
    core_ref, hypo_ref, dec_ref, groups = {}, {}, {}, {}
    for n, r in mv.items():
        if n not in dfj or dfj[n].get("vendor_hypo_ml") is None:
            continue
        cr = r.get("truth_core_ml") if yy == "18" else r.get("core_default")
        if cr in (None, "", "None"):
            continue
        core_ref[n], hypo_ref[n] = float(cr), float(dfj[n]["vendor_hypo_ml"])
        dec_ref[n] = r.get("defuse3_default") == "True"
        groups[n] = r.get("group") or n
    S = {"yy": yy, "reference_core": "DWI core" if yy == "18" else "vendor core", "methods": {}}
    for m in METHODS:
        full = grids(R, m, yy, "full")
        if not full:
            continue
        pats = sorted(n for n in full if n in core_ref)
        fold = folds({n: groups[n] for n in pats})
        lc = {n: np.log1p(full[n]["core_grid"]) for n in pats}
        lh = {n: np.log1p(full[n]["hypo_grid"]) for n in pats}
        choice = {}
        for k in range(5):
            tr = [n for n in pats if fold[n] != k]
            ec = np.median(np.abs(np.array([lc[n] for n in tr]) - np.log1p([core_ref[n] for n in tr])[:, None]), axis=0)
            eh = np.median(np.abs(np.array([lh[n] for n in tr]) - np.log1p([hypo_ref[n] for n in tr])[:, None]), axis=0)
            choice[k] = (int(np.argmin(ec)), int(np.argmin(eh)))
        block = {"n": len(pats), "chosen": sorted({str(v) for v in choice.values()})}
        for L in ["full", *a.lengths]:
            gl = full if L == "full" else grids(R, m, yy, f"T{L}")
            rows = []
            for n in pats:
                if n not in gl:
                    continue
                i, j = choice[fold[n]]
                c, h, u = gl[n]["core_grid"][i], gl[n]["hypo_grid"][j], gl[n]["union_grid"][i, j]
                cf, uf = full[n]["core_grid"][i], full[n]["union_grid"][i, j]
                rows.append({"ec": abs(np.log1p(c) - np.log1p(core_ref[n])), "ec_ml": abs(c - core_ref[n]),
                             "eh": abs(np.log1p(h) - np.log1p(hypo_ref[n])), "eh_ml": abs(h - hypo_ref[n]),
                             "dc": abs(c - cf), "du": abs(u - uf),
                             "flip": defuse3(c, u) != defuse3(cf, uf), "agree": defuse3(c, u) == dec_ref[n]})
            if not rows:
                continue
            med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
            block[str(L)] = {"n": len(rows), "core_err_log": med("ec"), "core_err_ml": med("ec_ml"),
                             "hypo_err_log": med("eh"), "hypo_err_ml": med("eh_ml"),
                             "core_change_ml": med("dc"), "union_change_ml": med("du"),
                             "flip": float(np.mean([r["flip"] for r in rows])),
                             "agree_vendor_decision": float(np.mean([r["agree"] for r in rows]))}
        S["methods"][m] = block
    text = json.dumps(S, indent=1)
    if a.out:
        a.out.write_text(text, encoding="utf8")
    # compact table
    keys = ["full", *map(str, a.lengths)]
    for metric in ("core_err_ml", "hypo_err_ml", "flip", "agree_vendor_decision"):
        if yy == "18" and metric in ("flip", "agree_vendor_decision"):
            continue
        print(f"\n{metric} ({S['reference_core'] if metric.startswith('core') else 'vendor Tmax>6' if metric.startswith('hypo') else 'DEFUSE 3'})")
        print("method      " + " ".join(f"{k:>7s}" for k in keys))
        for m, b in S["methods"].items():
            print(f"{m:11s} " + " ".join(f"{b[k][metric]:7.2f}" if k in b else "      -" for k in keys))


if __name__ == "__main__":
    main()
