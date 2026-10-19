"""Per-patient volumes of every estimator under truncation, for the figures of the paper.

python scripts/fig_per_patient_export.py <results_dir> <out.json> [--lengths 25 30 35 40 45]

Same cross-fitted thresholds as scripts/truncation_accuracy.py (chosen per method on the full
scans of the training folds) and the same fusions as scripts/weighted_fusion.py:
  DuoFusion w=0.5   core = learned + sSVD, log1p hypo = mean of twin and learned
  DuoFusion w       w cross-fitted on ISLES'24, fitted on all ISLES'24 for ISLES'18
Per data set (24, 18), per patient: references (core, hypo, DEFUSE 3 decision, group,
multiverse shares) and, per method and scan length, [core, hypo, union] in ml (union =
the cross-fitted core | hypo grid of the method; max(core, hypo) for the fusions).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truncation_accuracy as ta  # noqa: E402
from weighted_fusion import best_w, fused_hypo, references  # noqa: E402

MV_KEYS = ("defuse3_share", "dawn_share", "dawn_default", "core_min", "core_max", "core_default", "hypo_min",
           "hypo_max", "truth_core_ml")


def chosen(R, m, yy, lengths, core_ref, hypo_ref, groups):
    full = ta.grids(R, m, yy, "full")
    pats = sorted(n for n in full if n in core_ref)
    if not pats:
        return {}
    fold = ta.folds({n: groups[n] for n in pats})
    lc = {n: np.log1p(full[n]["core_grid"]) for n in pats}
    lh = {n: np.log1p(full[n]["hypo_grid"]) for n in pats}
    choice = {}
    for k in range(5):
        tr = [n for n in pats if fold[n] != k]
        ec = np.median(np.abs(np.array([lc[n] for n in tr]) - np.log1p([core_ref[n] for n in tr])[:, None]), axis=0)
        eh = np.median(np.abs(np.array([lh[n] for n in tr]) - np.log1p([hypo_ref[n] for n in tr])[:, None]), axis=0)
        choice[k] = (int(np.argmin(ec)), int(np.argmin(eh)))
    out = {}
    for L in ["full", *map(str, lengths)]:
        g = full if L == "full" else ta.grids(R, m, yy, f"T{L}")
        if not g:
            continue
        out[L] = {}
        for n in pats:
            if n in g:
                i, j = choice[fold[n]]
                out[L][n] = [float(g[n]["core_grid"][i]), float(g[n]["hypo_grid"][j]), float(g[n]["union_grid"][i, j])]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--lengths", type=int, nargs="+", default=[25, 30, 35, 40, 45])
    a = p.parse_args()
    R = a.results
    keys = ["full", *map(str, a.lengths)]
    data = {}
    for yy in ("24", "18"):
        core_ref, hypo_ref, dec_ref, groups = references(R, yy)
        vols = {m: chosen(R, m, yy, a.lengths, core_ref, hypo_ref, groups) for m in ta.METHODS}
        data[yy] = {"refs": (core_ref, hypo_ref, dec_ref, groups), "vols": vols}

    tw24, le24 = data["24"]["vols"]["duofusion"], data["24"]["vols"]["learned_ssvd"]
    core24, hypo24, _, g24 = data["24"]["refs"]
    pats24 = sorted(n for n in tw24["full"] if n in le24["full"])
    fold = ta.folds({n: g24[n] for n in pats24})
    pair = lambda d, L: {n: (d[L][n][0], d[L][n][1]) for n in d[L]}  # noqa: E731
    w_all = {L: best_w([n for n in pats24 if n in tw24.get(L, {}) and n in le24.get(L, {})],
                       pair(tw24, L), pair(le24, L), hypo24) for L in keys if L in tw24 and L in le24}
    out = {"lengths": keys, "w_all": w_all, "data": {}}
    for yy in ("24", "18"):
        core_ref, hypo_ref, dec_ref, groups = data[yy]["refs"]
        vols = data[yy]["vols"]
        tw, le = vols["duofusion"], vols["learned_ssvd"]
        f05, fw, wk_all = {}, {}, {}
        for L in keys:
            if L not in tw or L not in le:
                continue
            pats = [n for n in tw[L] if n in le[L]]
            f05[L], fw[L] = {}, {}
            if yy == "24":
                wk = {k: best_w([n for n in pats24 if fold[n] != k and n in tw[L] and n in le[L]],
                                pair(tw, L), pair(le, L), hypo_ref) for k in range(5)}
                wk_all[L] = wk
            for n in pats:
                c = le[L][n][0]
                h5 = fused_hypo(tw[L][n][1], le[L][n][1], 0.5)
                w = wk[fold[n]] if yy == "24" else w_all[L]
                hw = fused_hypo(tw[L][n][1], le[L][n][1], w)
                f05[L][n] = [c, h5, max(c, h5)]
                fw[L][n] = [c, hw, max(c, hw)]
        vols = {**vols, "duofusion_w05": f05, "duofusion_w": fw}
        import csv
        mvdir = R / ("mv24" if yy == "24" else "mv18")
        if not mvdir.exists():
            mvdir = R / f"duofusion_mv{yy}"
        mv = {int(r["patient"]): r for r in csv.DictReader(open(mvdir / "multiverse_per_patient.csv", encoding="utf8"))}
        out["data"][yy] = {
            "patients": {str(n): {"core_ref": core_ref[n], "hypo_ref": hypo_ref[n], "defuse3_ref": dec_ref[n],
                                  "group": str(groups[n]), **{k: mv[n].get(k) for k in MV_KEYS if n in mv}}
                         for n in core_ref},
            "volumes": {m: {L: {str(n): v for n, v in d.items()} for L, d in vl.items()} for m, vl in vols.items() if vl},
            "w_folds": {L: {str(k): v for k, v in wk.items()} for L, wk in wk_all.items()},
        }
    a.out.write_text(json.dumps(out))
    print({yy: {m: sorted(v) for m, v in out["data"][yy]["volumes"].items()} for yy in out["data"]})
    print("->", a.out, f"{a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
