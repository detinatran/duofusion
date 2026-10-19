"""How much do the core, hypoperfusion and DEFUSE 3 decision move when the scan is cut short?

python scripts/truncation_compare.py <results_dir> --lengths 30 40 [--out truncation_summary.json]

<results_dir> holds duofusion24/ (full scans) and duofusion24_T<L>/ (scripts/duofusion_volumes.py),
<method>24_full/ and <method>24_T<L>/ for method in osvd, ssvd, gamma_osvd
(scripts/osvd_volumes.py --method) and mv24/ (vendor volumes).
Each method's thresholds are fixed on the full scans, choosing on its grids the rCBF tau
and relative-Tmax d whose volumes best match the vendor core (rCBF < 0.3) and vendor
Tmax > 6 s volumes (smallest median |log1p ratio|). Then, per truncation length and
method, against the same method on the full scan of the same patient:
  core / hypo: median absolute change (ml), share of patients changing by > 10 ml
  DEFUSE 3 (core < 70 ml, ratio >= 1.8, mismatch >= 15 ml on the method's own volumes):
           share of decisions that flip
  DuoFusion only: share of full-scan volumes inside the 5-95% posterior range at length L
Only patients whose scan was actually cut at L enter the length-L numbers.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_grid(d, n):
    f = Path(d) / f"training_{n}.npz"
    return np.load(f) if f.exists() else None


def defuse3(core, union):
    hypo = union
    return bool(core < 70 and (hypo / core >= 1.8 if core > 0 else True) and hypo - core >= 15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--lengths", type=int, nargs="+", default=[30, 40])
    p.add_argument("--data", type=Path, default=None, help="data root with prep_info.json (frames per subject)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    R = a.results
    mv = {int(r["patient"]): r for r in csv.DictReader(open(R / "mv24" / "multiverse_per_patient.csv", encoding="utf8"))}
    dfj = {r["patient"]: r for r in json.load(open(R / "duofusion24" / "per_patient.json", encoding="utf8"))}
    pats = sorted(n for n in dfj if n in mv and dfj[n].get("vendor_hypo_ml") is not None)
    v_core = np.log1p([float(mv[n]["core_default"]) for n in pats])
    v_hypo = np.log1p([dfj[n]["vendor_hypo_ml"] for n in pats])
    methods = {"duofusion": ("duofusion24", "duofusion24_T"), "osvd": ("osvd24_full", "osvd24_T"),
               "ssvd": ("ssvd24_full", "ssvd24_T"), "gamma_osvd": ("gamma_osvd24_full", "gamma_osvd24_T")}
    S = {"n_patients": len(pats), "methods": {}}
    for m, (full_stem, trunc_stem) in methods.items():
        full_dir = R / full_stem
        if not full_dir.exists():
            continue
        g = {n: load_grid(full_dir, n) for n in pats}
        ok = [n for n in pats if g[n] is not None]
        idx = [pats.index(n) for n in ok]
        core = np.log1p(np.array([g[n]["core_grid"][0] if g[n]["core_grid"].ndim == 2 else g[n]["core_grid"] for n in ok]))
        hypo = np.log1p(np.array([g[n]["hypo_grid"][0] if g[n]["hypo_grid"].ndim == 2 else g[n]["hypo_grid"] for n in ok]))
        i_t = int(np.argmin(np.median(np.abs(core - v_core[idx, None]), axis=0)))
        j_d = int(np.argmin(np.median(np.abs(hypo - v_hypo[idx, None]), axis=0)))
        tau, dt = float(g[ok[0]]["tau_grid"][i_t]), float(g[ok[0]]["dt_grid"][j_d])
        block = {"tau": tau, "dt": dt, "n_full": len(ok),
                 "full_vs_vendor_core_mae_log": float(np.median(np.abs(core[:, i_t] - v_core[idx]))),
                 "full_vs_vendor_hypo_mae_log": float(np.median(np.abs(hypo[:, j_d] - v_hypo[idx])))}

        def point(z, key, i, j=None):
            arr = z[key]
            if m == "duofusion":
                arr = arr[0]
            return float(arr[i] if j is None else arr[i, j])

        for L in a.lengths:
            gl = {n: load_grid(R / f"{trunc_stem}{L}", n) for n in ok}
            cut = [n for n in ok if gl[n] is not None]
            if a.data is not None:
                keep = []
                for n in cut:
                    info = json.loads((a.data / f"training_{n}" / "prep_info.json").read_text(encoding="utf8"))
                    if info.get("frames", 0) > L + 1:
                        keep.append(n)
                cut = keep
            dc, dh, flips, cov_c, cov_h = [], [], [], [], []
            for n in cut:
                cf, hf = point(g[n], "core_grid", i_t), point(g[n], "hypo_grid", j_d)
                cl, hl = point(gl[n], "core_grid", i_t), point(gl[n], "hypo_grid", j_d)
                uf, ul = point(g[n], "union_grid", i_t, j_d), point(gl[n], "union_grid", i_t, j_d)
                dc.append(cl - cf)
                dh.append(hl - hf)
                flips.append(defuse3(cf, uf) != defuse3(cl, ul))
                if m == "duofusion":
                    dcore = gl[n]["core_grid"][1:, i_t]
                    dhyp = gl[n]["hypo_grid"][1:, j_d]
                    cov_c.append(np.percentile(dcore, 5) <= cf <= np.percentile(dcore, 95))
                    cov_h.append(np.percentile(dhyp, 5) <= hf <= np.percentile(dhyp, 95))
            dc, dh = np.array(dc), np.array(dh)
            res = {"n": len(cut),
                   "core_abs_change_ml_median": float(np.median(np.abs(dc))) if len(dc) else None,
                   "core_signed_change_ml_median": float(np.median(dc)) if len(dc) else None,
                   "core_change_gt10ml": float(np.mean(np.abs(dc) > 10)) if len(dc) else None,
                   "hypo_abs_change_ml_median": float(np.median(np.abs(dh))) if len(dh) else None,
                   "hypo_signed_change_ml_median": float(np.median(dh)) if len(dh) else None,
                   "hypo_change_gt10ml": float(np.mean(np.abs(dh) > 10)) if len(dh) else None,
                   "defuse3_flip": float(np.mean(flips)) if flips else None}
            if cov_c:
                res["posterior_covers_full_core"] = float(np.mean(cov_c))
                res["posterior_covers_full_hypo"] = float(np.mean(cov_h))
            block[f"T{L}"] = res
        S["methods"][m] = block
    text = json.dumps(S, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf8")


if __name__ == "__main__":
    main()
