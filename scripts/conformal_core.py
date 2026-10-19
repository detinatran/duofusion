"""Calibrated core-volume intervals (ISLES 2018) and eligibility probabilities (ISLES'24).

python scripts/conformal_core.py <mv18_dir> <mv24_dir> <out_dir> [--repeats 20]

Input: multiverse_per_patient.csv of scripts/multiverse_eligibility.py for ISLES 2018
(label = core on DWI right after the CTP; "group" = patient) and ISLES'24.

Intervals for the ISLES 2018 core volume, evaluated by 5-fold cross-validation over
patients, repeated with different splits (all volumes on the log1p scale):
  range      [min, max] of the 90 analysis variants (no calibration)
  residual   default core + conformal quantiles of the signed residual
             log1p(truth) - log1p(default), from the training folds (the predictor
             is not fitted, so the training residuals are exchangeable with the test ones)
  cqr        conformalised quantile regression: linear quantile regression of log1p(truth)
             on log1p of the default/quartile/extreme core and of the hypoperfused volume,
             fitted on 60% of the training patients and calibrated on the other 40%
Coverage and median width (ml) at 80% and 90% nominal.

ISLES'24: the residual distribution of all ISLES 2018 slabs turns each default core into
a predictive distribution, core_j = expm1(log1p(core) + e_j); the share of draws that meet
DEFUSE 3 (with the default hypoperfused volume) or DAWN is the eligibility probability.
This transfer assumes that the vendor-map error of ISLES 2018 holds for ISLES'24 (other
centres and software, whole brain instead of slabs): exploratory.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import QuantileRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiverse_eligibility import dawn, defuse3  # noqa: E402

FEATS = ("core_default", "core_q25", "core_median", "core_q75", "core_max", "hypo_default")
ALPHAS = (0.2, 0.1)


def read(path):
    rows = list(csv.DictReader(open(path, encoding="utf8")))
    for r in rows:
        for k, v in r.items():
            if v in ("", "None"):
                r[k] = None
            elif k not in ("defuse3_default", "dawn_default"):
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def cq(scores, alpha):
    """Conformal quantile with the finite-sample correction."""
    n = len(scores)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(scores)[k - 1])


def group_folds(groups, k, rng):
    keys = np.array(sorted(set(groups)))
    rng.shuffle(keys)
    fold_of = {g: i % k for i, g in enumerate(keys)}
    return np.array([fold_of[g] for g in groups])


def evaluate(rows, repeats, seed=0):
    y = np.log1p([r["truth_core_ml"] for r in rows])
    d = np.log1p([r["core_default"] for r in rows])
    lo_r, hi_r = np.log1p([r["core_min"] for r in rows]), np.log1p([r["core_max"] for r in rows])
    X = np.log1p(np.array([[r[f] for f in FEATS] for r in rows]))
    groups = [r["group"] for r in rows]
    res = {"range": {"coverage": float(np.mean((lo_r <= y) & (y <= hi_r))),
                     "width_ml_median": float(np.median(np.expm1(hi_r) - np.expm1(lo_r)))}}
    for alpha in ALPHAS:
        cov = {"residual": [], "cqr": []}
        wid = {"residual": [], "cqr": []}
        for rep in range(repeats):
            rng = np.random.default_rng(seed + rep)
            fold = group_folds(groups, 5, rng)
            for k in range(5):
                te, tr = fold == k, fold != k
                # residual conformal (two one-sided quantiles)
                e = y[tr] - d[tr]
                lo = d[te] - cq(-e, alpha / 2)
                hi = d[te] + cq(e, alpha / 2)
                cov["residual"] += list((lo <= y[te]) & (y[te] <= hi))
                wid["residual"] += list(np.expm1(hi) - np.expm1(np.maximum(lo, 0)))
                # CQR: proper training / calibration split over patients
                tr_idx = np.nonzero(tr)[0]
                gtr = np.array(groups)[tr_idx]
                inner = group_folds(list(gtr), 5, rng)
                fit, cal = tr_idx[inner >= 2], tr_idx[inner < 2]
                qs = []
                for q in (alpha / 2, 1 - alpha / 2):
                    m = QuantileRegressor(quantile=q, alpha=1e-3, solver="highs").fit(X[fit], y[fit])
                    qs.append(m)
                s = np.maximum(qs[0].predict(X[cal]) - y[cal], y[cal] - qs[1].predict(X[cal]))
                qhat = cq(s, alpha)
                lo = qs[0].predict(X[te]) - qhat
                hi = qs[1].predict(X[te]) + qhat
                cov["cqr"] += list((lo <= y[te]) & (y[te] <= hi))
                wid["cqr"] += list(np.expm1(hi) - np.expm1(np.maximum(lo, 0)))
        for m in cov:
            res.setdefault(m, {})[f"coverage_{int(100 * (1 - alpha))}"] = float(np.mean(cov[m]))
            res[m][f"width_ml_median_{int(100 * (1 - alpha))}"] = float(np.median(wid[m]))
    return res


def eligibility(rows24, residuals):
    out = []
    for r in rows24:
        core = np.expm1(np.log1p(r["core_default"]) + residuals)
        hyp = max(r["hypo_default"], 0.0)
        p3 = float(np.mean([defuse3(c, max(hyp, c)) for c in core]))
        dv = [dawn(c, r["age"], r["nihss"]) for c in core]
        pd = float(np.mean(dv)) if dv and dv[0] is not None else None
        out.append({"patient": int(r["patient"]), "core_default": r["core_default"], "hypo_default": hyp,
                    "core_p10": float(np.percentile(core, 10)), "core_p50": float(np.median(core)),
                    "core_p90": float(np.percentile(core, 90)),
                    "defuse3_default": r["defuse3_default"] == "True", "p_defuse3": p3,
                    "defuse3_multiverse_share": r["defuse3_share"],
                    "dawn_default": r["dawn_default"], "p_dawn": pd, "dawn_multiverse_share": r["dawn_share"]})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mv18", type=Path)
    p.add_argument("mv24", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--repeats", type=int, default=20)
    a = p.parse_args()
    r18 = [r for r in read(a.mv18 / "multiverse_per_patient.csv") if r.get("truth_core_ml") is not None]
    r24 = read(a.mv24 / "multiverse_per_patient.csv")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"isles18": {"n_slabs": len(r18), "n_patients": len({r["group"] for r in r18}),
                           **evaluate(r18, a.repeats)}}
    e = np.log1p([r["truth_core_ml"] for r in r18]) - np.log1p([r["core_default"] for r in r18])
    summary["isles18"]["residual_log1p"] = {"median": float(np.median(e)), "q10": float(np.percentile(e, 10)),
                                            "q90": float(np.percentile(e, 90))}
    el = eligibility(r24, e)
    with open(a.out_dir / "eligibility_isles24.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(el[0]))
        w.writeheader()
        w.writerows(el)

    def band(v):
        v = [x for x in v if x is not None]
        return {"n": len(v), "certain_no": int(sum(x <= 0.1 for x in v)), "uncertain": int(sum(0.1 < x < 0.9 for x in v)),
                "certain_yes": int(sum(x >= 0.9 for x in v))}

    summary["isles24"] = {"n": len(el), "defuse3_eligible_default": int(sum(x["defuse3_default"] for x in el)),
                          "p_defuse3": band([x["p_defuse3"] for x in el]),
                          "p_dawn": band([x["p_dawn"] for x in el]),
                          "core_default_median": float(np.median([x["core_default"] for x in el])),
                          "core_p50_median": float(np.median([x["core_p50"] for x in el]))}
    # which DEFUSE 3 component varies for patients unstable in the multiverse
    unstable = [r for r in r24 if 0.1 < r["defuse3_share"] < 0.9]
    summary["isles24"]["multiverse_unstable_defuse3"] = {
        "n": len(unstable),
        "core_lt70_varies": int(sum(0 < r["d3_core_lt70_share"] < 1 for r in unstable)),
        "ratio_varies": int(sum(0 < r["d3_ratio_ge18_share"] < 1 for r in unstable)),
        "mismatch_ge15_varies": int(sum(0 < r["d3_mismatch_ge15_share"] < 1 for r in unstable))}
    (a.out_dir / "conformal_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
