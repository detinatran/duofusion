"""Quick test: does fusing the twin with the learned extrapolation beat both under truncation?

python scripts/fusion_test.py <results_dir> --yy 24 --lengths 25 30 35

Volume-level fusion with the cross-fitted thresholds of scripts/truncation_accuracy.py
(each method keeps its own tau and d, chosen on the full scans of the training folds):
  F_hypo     core = learned, hypo = geometric mean(twin, learned)
  F_both     core = geometric mean(twin, learned), hypo = geometric mean(twin, learned)
  F_twinhyp  core = learned, hypo = twin
The DEFUSE 3 union is approximated by max(core, hypo) for the fused variants (and for the
two single methods, so that all rows are comparable). Errors against the same references
as truncation_accuracy.py; flips against the variant's own full scan.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truncation_accuracy as ta  # noqa: E402


def chosen_volumes(R, m, yy, lengths, core_ref, hypo_ref, groups):
    full = ta.grids(R, m, yy, "full")
    pats = sorted(n for n in full if n in core_ref)
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
        out[L] = {}
        for n in pats:
            if n in g:
                i, j = choice[fold[n]]
                out[L][n] = (float(g[n]["core_grid"][i]), float(g[n]["hypo_grid"][j]))
    return out


def gm(a, b):
    return float(np.expm1(0.5 * (np.log1p(a) + np.log1p(b))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--yy", default="24")
    p.add_argument("--lengths", type=int, nargs="+", default=[25, 30, 35])
    a = p.parse_args()
    R, yy = a.results, a.yy
    import csv
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
    tw = chosen_volumes(R, "duofusion", yy, a.lengths, core_ref, hypo_ref, groups)
    le = chosen_volumes(R, "learned_ssvd", yy, a.lengths, core_ref, hypo_ref, groups)
    ga = chosen_volumes(R, "gamma_ssvd", yy, a.lengths, core_ref, hypo_ref, groups)
    variants = {
        "twin": lambda t, l: t,
        "learned": lambda t, l: l,
        "F_hypo": lambda t, l: (l[0], gm(t[1], l[1])),
        "F_both": lambda t, l: (gm(t[0], l[0]), gm(t[1], l[1])),
        "F_twinhyp": lambda t, l: (l[0], t[1]),
    }
    rows = {}
    keys = ["full", *map(str, a.lengths)]
    for name, f in [*variants.items(), ("gamma_ssvd", None)]:
        res = {}
        src = ga if name == "gamma_ssvd" else None
        for L in keys:
            pats = [n for n in core_ref if n in tw[L] and n in le[L] and n in tw["full"] and n in le["full"]]
            m = {"ec": [], "eh": [], "flip": [], "agree": []}
            for n in pats:
                if src is not None:
                    if n not in src[L] or n not in src["full"]:
                        continue
                    c, h = src[L][n]
                    cf, hf = src["full"][n]
                else:
                    c, h = f(tw[L][n], le[L][n])
                    cf, hf = f(tw["full"][n], le["full"][n])
                m["ec"].append(abs(c - core_ref[n]))
                m["eh"].append(abs(h - hypo_ref[n]))
                m["flip"].append(ta.defuse3(c, max(c, h)) != ta.defuse3(cf, max(cf, hf)))
                m["agree"].append(ta.defuse3(c, max(c, h)) == dec_ref[n])
            res[L] = {k: float(np.median(v)) if k in ("ec", "eh") else float(np.mean(v)) for k, v in m.items()}
        rows[name] = res
    for metric, label in (("ec", "core |err| ml"), ("eh", "hypo |err| ml"), ("flip", "DEFUSE 3 flip"),
                          ("agree", "agree with vendor decision")):
        if yy == "18" and metric in ("flip", "agree"):
            continue
        print(f"\n{label} (ISLES'{yy})")
        print("variant     " + " ".join(f"{k:>7s}" for k in keys))
        for name, res in rows.items():
            print(f"{name:11s} " + " ".join(f"{res[k][metric]:7.2f}" for k in keys))


if __name__ == "__main__":
    main()
