"""Hybrid of the truncation study: core from sSVD, hypoperfusion from the twin's relative Tmax.

python scripts/hybrid_decision.py <results_dir> <data_root> --lengths 25 30 35 40 45 [--tau 0.35] [--dt 2.0]

Reads the saved maps of scripts/osvd_volumes.py --method ssvd (SSVD_CBF, SSVD_Tmax_rel in
ssvd24_<full|T L>/training_X/) and of scripts/duofusion_volumes.py (DF_Tmax_rel in
duofusion24[_T L]/training_X/). Per scan length:
  core   = sSVD rCBF < tau, reference = sSVD CBF median of the healthy contralateral tissue
           (twin Tmax_rel < 2 s, side = hemisphere with more twin Tmax_rel > 4 s)
  hypo   = twin Tmax_rel > dt, union = core | hypo (voxel sets)
and the DEFUSE 3 decision on (core, union). Reports, against the same hybrid on the full
scan: median |change| of core and union volume, share of decisions that flip; and on the
full scan the agreement with the vendor core and Tmax > 6 s volumes.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from duofusion.data import load_subject  # noqa: E402
from duofusion.reference import affected_side  # noqa: E402


def defuse3(core, union):
    return bool(core < 70 and (union / core >= 1.8 if core > 0 else True) and union - core >= 15)


def volumes(res, data_root, n, suf, tau, dt):
    sd = res / f"ssvd24_{suf}" / f"training_{n}"
    dd = res / ("duofusion24" if suf == "full" else f"duofusion24_{suf}") / f"training_{n}"
    if not ((sd / "SSVD_CBF.nii.gz").exists() and (dd / "DF_Tmax_rel.nii.gz").exists()):
        return None
    s = load_subject(data_root, n)
    cbf = nib.load(str(sd / "SSVD_CBF.nii.gz")).get_fdata()
    trel = nib.load(str(dd / "DF_Tmax_rel.nii.gz")).get_fdata()
    valid = (s.tissue > 0) & np.isfinite(cbf) & (cbf > 0) & np.isfinite(trel)
    side = affected_side(s, valid & (trel > 4.0))
    contra = (s.hemi == 3 - side) if side else (s.hemi > 0)
    healthy = valid & (trel < 2.0)
    sel = healthy & contra if np.any(healthy & contra) else (healthy if np.any(healthy) else valid)
    r = cbf / float(np.median(cbf[sel]))
    core = valid & (r < tau)
    union = core | (valid & (trel > dt))
    return float(core.sum()) * s.voxel_ml, float(union.sum()) * s.voxel_ml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("data_root", type=Path)
    p.add_argument("--lengths", type=int, nargs="+", default=[25, 30, 35, 40, 45])
    p.add_argument("--tau", type=float, default=0.35)
    p.add_argument("--dt", type=float, default=2.0)
    a = p.parse_args()
    mv = {int(r["patient"]): r for r in csv.DictReader(open(a.results / "mv24" / "multiverse_per_patient.csv", encoding="utf8"))}
    dfj = {r["patient"]: r for r in json.load(open(a.results / "duofusion24" / "per_patient.json", encoding="utf8"))}
    pats = sorted(n for n in dfj if n in mv)
    full = {n: volumes(a.results, a.data_root, n, "full", a.tau, a.dt) for n in pats}
    pats = [n for n in pats if full[n] is not None]
    vc = np.log1p([float(mv[n]["core_default"]) for n in pats])
    vh = np.log1p([dfj[n]["vendor_hypo_ml"] for n in pats])
    S = {"tau": a.tau, "dt": a.dt, "n": len(pats),
         "full_vs_vendor_core_mae_log": float(np.median(np.abs(np.log1p([full[n][0] for n in pats]) - vc))),
         "full_vs_vendor_union_mae_log": float(np.median(np.abs(np.log1p([full[n][1] for n in pats]) - vh)))}
    for L in a.lengths:
        dc, du, flips = [], [], []
        for n in pats:
            v = volumes(a.results, a.data_root, n, f"T{L}", a.tau, a.dt)
            if v is None:
                continue
            dc.append(v[0] - full[n][0])
            du.append(v[1] - full[n][1])
            flips.append(defuse3(*v) != defuse3(*full[n]))
        if flips:
            S[f"T{L}"] = {"n": len(flips), "core_abs_change_ml_median": float(np.median(np.abs(dc))),
                          "union_abs_change_ml_median": float(np.median(np.abs(du))),
                          "defuse3_flip": float(np.mean(flips))}
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
