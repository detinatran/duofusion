"""How fragile is the thrombectomy eligibility decision to defensible analysis choices?

python scripts/multiverse_eligibility.py <data_root> <out_dir> [--workers 8] [--truth]

For every patient, core (rCBF < tau) and hypoperfusion (Tmax > t) volumes are computed
from the vendor maps under every combination of
  reference  contra_hemi (RAPID-like), brain_median, healthy_tissue
  tau        0.20, 0.25, 0.30, 0.34, 0.38
  tmax       5.5, 6.0, 6.5 s   (about the frame-level uncertainty of Tmax)
  smoothing  none, 2 mm Gaussian on CBF and Tmax
and the decision is evaluated with
  DEFUSE 3   core < 70 ml, mismatch ratio >= 1.8, mismatch volume >= 15 ml
  DAWN       NIHSS >= 10 and core < 21 ml (age >= 80), core < 31 ml (age < 80),
             or NIHSS >= 20, age < 80 and core < 51 ml   (needs age and NIHSS)
The default variant is contra_hemi, 0.30, 6.0 s, no smoothing. A patient is "unstable"
when the share of eligible variants is strictly between 10% and 90%.
With --truth (ISLES 2018: label = core on DWI right after the CTP) the coverage of the
core volume by the multiverse range is reported as well (slab volumes).
"""
import argparse
import csv
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from duofusion.data import list_subjects, load_subject  # noqa: E402
from duofusion.reference import classic_references, masks, nan_gaussian, spacing_of  # noqa: E402

REFS = ("contra_hemi", "brain_median", "healthy_tissue")
TAUS = (0.20, 0.25, 0.30, 0.34, 0.38)
TMAXS = (5.5, 6.0, 6.5)
SMOOTH = (0.0, 2.0)
DEFAULT = ("contra_hemi", 0.30, 6.0, 0.0)


def defuse3(core, hypo):
    mismatch = hypo - core
    ratio = hypo / core if core > 0 else np.inf
    return core < 70 and ratio >= 1.8 and mismatch >= 15


def dawn(core, age, nihss):
    if age is None or nihss is None or not np.isfinite(age) or not np.isfinite(nihss):
        return None
    if nihss >= 10 and ((age >= 80 and core < 21) or (age < 80 and core < 31)):
        return True
    return bool(nihss >= 20 and age < 80 and core < 51)


def work(args):
    root, num = args
    s = load_subject(root, num)
    cbf0, tmax0 = s.maps["CBF"], s.maps["Tmax"]
    valid, _, _ = masks(s, cbf0, tmax0)
    sp = spacing_of(s.affine)
    age = s.info.get("age")
    nihss = s.info.get("nihss_admission")
    age = float(age) if age is not None else None
    nihss = float(nihss) if nihss is not None else None
    rows = []
    for sig in SMOOTH:
        if sig:
            cbf = np.where(valid, nan_gaussian(np.where(valid, cbf0, np.nan), sig, sp), np.nan)
            tmax = np.where(valid, nan_gaussian(np.where(valid & (tmax0 >= 0), tmax0, np.nan), sig, sp), np.nan)
        else:
            cbf, tmax = cbf0, tmax0
        refs, _ = classic_references(s, cbf, tmax)
        for ref in REFS:
            with np.errstate(invalid="ignore", divide="ignore"):
                r = cbf / refs[ref]
            for tau, tt in itertools.product(TAUS, TMAXS):
                with np.errstate(invalid="ignore"):
                    core = valid & (r < tau)
                    hypo = valid & (tmax > tt)
                core_ml, hypo_ml = float(core.sum()) * s.voxel_ml, float((hypo | core).sum()) * s.voxel_ml
                rows.append({"ref": ref, "tau": tau, "tmax": tt, "smooth": sig, "core_ml": core_ml,
                             "hypo_ml": hypo_ml, "defuse3": defuse3(core_ml, hypo_ml), "dawn": dawn(core_ml, age, nihss)})
    d = next(r for r in rows if (r["ref"], r["tau"], r["tmax"], r["smooth"]) == DEFAULT)
    core = np.array([r["core_ml"] for r in rows])
    hyp = np.array([r["hypo_ml"] for r in rows])
    group = s.info.get("patient")
    out = {"patient": num, "group": group if group is not None else num, "age": age, "nihss": nihss,
           "brain_ml": float(valid.sum()) * s.voxel_ml,
           "core_default": d["core_ml"], "hypo_default": d["hypo_ml"],
           "core_min": float(core.min()), "core_q25": float(np.percentile(core, 25)),
           "core_median": float(np.median(core)),
           "core_q75": float(np.percentile(core, 75)), "core_max": float(core.max()),
           "hypo_min": float(hyp.min()), "hypo_max": float(hyp.max()),
           # which DEFUSE 3 component varies across the multiverse
           "d3_core_lt70_share": float(np.mean(core < 70)),
           "d3_ratio_ge18_share": float(np.mean([h / c >= 1.8 if c > 0 else True for c, h in zip(core, hyp)])),
           "d3_mismatch_ge15_share": float(np.mean(hyp - core >= 15)),
           "defuse3_default": d["defuse3"], "defuse3_share": float(np.mean([r["defuse3"] for r in rows]))}
    dv = [r["dawn"] for r in rows if r["dawn"] is not None]
    out["dawn_default"] = d["dawn"]
    out["dawn_share"] = float(np.mean(dv)) if dv else None
    if s.ot.any():
        out["truth_core_ml"] = float((s.ot & valid).sum()) * s.voxel_ml
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--truth", action="store_true", help="label is the core at CTP time (ISLES 2018)")
    a = p.parse_args()
    nums = list_subjects(a.data_root)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(work, [(a.data_root, n) for n in nums]))
    with open(a.out_dir / "multiverse_per_patient.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k != "patient", k)))
        w.writeheader()
        w.writerows(rows)

    def unstable(sh):
        return sh is not None and 0.1 < sh < 0.9

    def flips(sh):
        return sh is not None and 0.0 < sh < 1.0

    S = {"n": len(rows), "variants": len(REFS) * len(TAUS) * len(TMAXS) * len(SMOOTH),
         "brain_ml_median": float(np.median([r["brain_ml"] for r in rows]))}
    for crit in ("defuse3", "dawn"):
        sh = [r[f"{crit}_share"] for r in rows if r[f"{crit}_share"] is not None]
        if not sh:
            continue
        S[crit] = {"n": len(sh), "eligible_default": int(sum(bool(r[f"{crit}_default"]) for r in rows
                                                              if r[f"{crit}_default"] is not None)),
                   "any_flip": int(sum(flips(x) for x in sh)), "unstable_10_90": int(sum(unstable(x) for x in sh))}
    spread = np.array([r["core_max"] - r["core_min"] for r in rows])
    iqr = np.array([r["core_q75"] - r["core_q25"] for r in rows])
    S["core_range_ml"] = {"median": float(np.median(spread)), "q75": float(np.percentile(spread, 75))}
    S["core_iqr_ml"] = {"median": float(np.median(iqr)), "q75": float(np.percentile(iqr, 75))}
    S["core_default_ml_median"] = float(np.median([r["core_default"] for r in rows]))
    if a.truth:
        t = [r for r in rows if "truth_core_ml" in r]
        cov = [r["core_min"] <= r["truth_core_ml"] <= r["core_max"] for r in t]
        cov_iqr = [r["core_q25"] <= r["truth_core_ml"] <= r["core_q75"] for r in t]
        err = [r["core_default"] - r["truth_core_ml"] for r in t]
        S["truth"] = {"n": len(t), "range_covers": float(np.mean(cov)), "iqr_covers": float(np.mean(cov_iqr)),
                      "default_err_median_ml": float(np.median(err)),
                      "default_abs_err_median_ml": float(np.median(np.abs(err))),
                      "truth_below_min": float(np.mean([r["truth_core_ml"] < r["core_min"] for r in t])),
                      "truth_above_max": float(np.mean([r["truth_core_ml"] > r["core_max"] for r in t]))}
    (a.out_dir / "multiverse_summary.json").write_text(json.dumps(S, indent=1), encoding="utf8")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
