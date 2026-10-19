"""Weighted fusion of the twin and the learned extrapolation, weight learned on ISLES'24
and tested on ISLES 2018 unchanged.

python scripts/weighted_fusion.py <results_dir> --lengths 25 30 35

Core = learned extrapolation + sSVD (better than the twin's core on both data sets).
Hypoperfusion, on the log1p scale: log h = w log h_twin + (1 - w) log h_learned, with one
weight per scan length (full scan included), w on a 0..1 grid of 0.1, chosen to minimise the
median |log1p error| against the vendor Tmax > 6 s volume:
  ISLES'24  cross-fitted: w from the training folds, applied to the test fold
  ISLES'18  external: w fitted on all ISLES'24 patients, applied unchanged
Compared with the single methods (twin, learned, gamma + sSVD) and the fixed fusions
(w = 0.5, w = 1). Metrics as scripts/truncation_accuracy.py (DEFUSE 3 union = max(core, hypo)).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truncation_accuracy as ta  # noqa: E402
from fusion_test import chosen_volumes  # noqa: E402

W_GRID = np.round(np.arange(0.0, 1.001, 0.1), 2)


def references(R, yy):
    mvdir = R / ("mv24" if yy == "24" else "mv18")
    if not mvdir.exists():
        mvdir = R / f"duofusion_mv{yy}"
    mv = {int(r["patient"]): r for r in csv.DictReader(open(mvdir / "multiverse_per_patient.csv", encoding="utf8"))}
    dfj = {r["patient"]: r for r in json.load(open(R / f"duofusion{yy}" / "per_patient.json", encoding="utf8"))}
    core_ref, hypo_ref, dec_ref, groups = {}, {}, {}, {}
    for n, r in mv.items():
        cr = r.get("truth_core_ml") if yy == "18" else r.get("core_default")
        if n not in dfj or dfj[n].get("vendor_hypo_ml") is None or cr in (None, "", "None"):
            continue
        core_ref[n], hypo_ref[n] = float(cr), float(dfj[n]["vendor_hypo_ml"])
        dec_ref[n] = r.get("defuse3_default") == "True"
        groups[n] = r.get("group") or n
    return core_ref, hypo_ref, dec_ref, groups


def fused_hypo(ht, hl, w):
    return float(np.expm1(w * np.log1p(ht) + (1 - w) * np.log1p(hl)))


def best_w(pats, tw, le, hypo_ref):
    err = [np.median([abs(np.log1p(fused_hypo(tw[n][1], le[n][1], w)) - np.log1p(hypo_ref[n])) for n in pats])
           for w in W_GRID]
    return float(W_GRID[int(np.argmin(err))])


def evaluate(volumes, core_ref, hypo_ref, dec_ref, keys):
    """volumes[L][n] = (core, hypo). Returns per-L median errors, flips vs full, agreement."""
    out = {}
    for L in keys:
        pats = [n for n in volumes[L] if n in volumes["full"]]
        ec = [abs(volumes[L][n][0] - core_ref[n]) for n in pats]
        eh = [abs(volumes[L][n][1] - hypo_ref[n]) for n in pats]
        dec = lambda c, h: ta.defuse3(c, max(c, h))  # noqa: E731
        flip = [dec(*volumes[L][n]) != dec(*volumes["full"][n]) for n in pats]
        agree = [dec(*volumes[L][n]) == dec_ref[n] for n in pats]
        out[L] = {"ec": float(np.median(ec)), "eh": float(np.median(eh)),
                  "flip": float(np.mean(flip)), "agree": float(np.mean(agree))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--lengths", type=int, nargs="+", default=[25, 30, 35])
    a = p.parse_args()
    R = a.results
    keys = ["full", *map(str, a.lengths)]
    data = {}
    for yy in ("24", "18"):
        refs = references(R, yy)
        data[yy] = {"refs": refs, **{m: chosen_volumes(R, m, yy, a.lengths, *refs[:2], refs[3])
                                     for m in ("duofusion", "learned_ssvd", "gamma_ssvd")}}
    # weights: cross-fitted on ISLES'24, and fitted on all ISLES'24 for the external test
    tw24, le24 = data["24"]["duofusion"], data["24"]["learned_ssvd"]
    g24 = data["24"]["refs"][3]
    pats24 = sorted(n for n in tw24["full"] if n in le24["full"])
    fold = ta.folds({n: g24[n] for n in pats24})
    w_all = {L: best_w([n for n in pats24 if n in tw24[L] and n in le24[L]], tw24[L], le24[L], data["24"]["refs"][1])
             for L in keys}
    print("weights on the twin (fitted on all ISLES'24):", w_all)
    for yy in ("24", "18"):
        core_ref, hypo_ref, dec_ref, _ = data[yy]["refs"]
        tw, le, ga = data[yy]["duofusion"], data[yy]["learned_ssvd"], data[yy]["gamma_ssvd"]
        vols = {"twin": {}, "learned": {}, "gamma_ssvd": {}, "fuse_w0.5": {}, "learned_core+twin_hypo": {},
                "DuoFusion (weighted)": {}}
        for L in keys:
            pats = [n for n in tw[L] if n in le[L]]
            vols["twin"][L] = {n: tw[L][n] for n in pats}
            vols["learned"][L] = {n: le[L][n] for n in pats}
            vols["gamma_ssvd"][L] = {n: ga[L][n] for n in pats if n in ga.get(L, {})}
            vols["fuse_w0.5"][L] = {n: (le[L][n][0], fused_hypo(tw[L][n][1], le[L][n][1], 0.5)) for n in pats}
            vols["learned_core+twin_hypo"][L] = {n: (le[L][n][0], tw[L][n][1]) for n in pats}
            if yy == "24":
                wk = {k: best_w([n for n in pats24 if fold[n] != k and n in tw[L] and n in le[L]], tw[L], le[L], hypo_ref)
                      for k in range(5)}
                vols["DuoFusion (weighted)"][L] = {n: (le[L][n][0], fused_hypo(tw[L][n][1], le[L][n][1], wk[fold[n]]))
                                                   for n in pats}
            else:
                vols["DuoFusion (weighted)"][L] = {n: (le[L][n][0], fused_hypo(tw[L][n][1], le[L][n][1], w_all[L]))
                                                   for n in pats}
        res = {name: evaluate(v, core_ref, hypo_ref, dec_ref, keys) for name, v in vols.items()}
        for metric, label in (("eh", "hypo |err| ml"), ("ec", "core |err| ml"), ("flip", "DEFUSE 3 flip"),
                              ("agree", "agree with vendor decision")):
            if yy == "18" and metric in ("flip", "agree"):
                continue
            print(f"\n{label} (ISLES'{yy}{', external, weights from ISLES 24' if yy == '18' else ', cross-fitted'})")
            print("variant                   " + " ".join(f"{k:>7s}" for k in keys))
            for name, r in res.items():
                print(f"{name:25s} " + " ".join(f"{r[k][metric]:7.2f}" for k in keys))


if __name__ == "__main__":
    main()
