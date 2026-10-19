"""Cohort summary of the uncertainty results (run after run_uq.py).

python scripts/eval_uq.py configs/full.yaml --data-root ../HemoPIC/data/ISLES2017_Training

1. Identifiability: for every parameter, how much the data narrowed it
   (posterior sd / prior sd; ~1 = the prior alone decides, < 0.5 = learnt from
   the data) and how often it sits at a bound.
2. Size of the uncertainty: relative 95% width of CBF, V, MTT, tau by tissue;
   MCMC convergence.
3. Check 2, uncertainty vs lesion detection: lesion ROC AUC after removing the
   q% most uncertain voxels, compared with removing q% at random.
4. Is the lesion more uncertain than the contralateral side?
5. The paper's tau claim (lesion relaxes faster than contralateral tissue) with
   posterior uncertainty: weighted cohort median tau of lesion-overlapping vs
   contralateral-overlapping ROIs, per posterior draw.

Writes <out_root>/<name>/eval_uq/summary.json and prints a report.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, subject_id  # noqa: E402

FRACS = (0.0, 0.1, 0.2, 0.3, 0.4)
PARAMS = ("log_R", "log_tau", "log_V", "log_F", "log_Cin", "delay", "log_disp", "log_Pe")


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def selective(data_dir, fit_dir, rng):
    """AUC after removing the most uncertain voxels vs removing random voxels."""
    seg = nib.load(str(data_dir / SEG4_REL)).get_fdata() > 0
    ot = nib.load(str(find_nii(data_dir, "OT"))).get_fdata() > 0
    out = {}
    for m, direction in (("CBF", -1.0), ("MTT", 1.0)):
        val = nib.load(str(fit_dir / f"HemoPIC_{m}.nii.gz")).get_fdata()
        lo = nib.load(str(fit_dir / "uq" / f"HemoPIC_{m}_lo.nii.gz")).get_fdata()
        hi = nib.load(str(fit_dir / "uq" / f"HemoPIC_{m}_hi.nii.gz")).get_fdata()
        ok = seg & np.isfinite(val) & (val > 0) & np.isfinite(lo) & np.isfinite(hi)
        if ot[ok].sum() < 10:
            continue
        y = ot[ok].astype(np.uint8)
        s = direction * val[ok]
        u = (hi[ok] - lo[ok]) / val[ok] + 1e-9 * rng.random(int(ok.sum()))
        order = np.argsort(-u)
        res_u, res_r = [], []
        for q in FRACS:
            k = int(round(q * y.size))
            keep = np.ones(y.size, bool)
            keep[order[:k]] = False
            res_u.append(roc_auc(y[keep], s[keep]))
            rr = []
            for _ in range(10):
                keep = np.ones(y.size, bool)
                keep[rng.choice(y.size, size=k, replace=False)] = False
                rr.append(roc_auc(y[keep], s[keep]))
            res_r.append(float(np.nanmean(rr)))
        flip = np.flip(ot, axis=0) & ~ot
        u_full = np.where(ok, (hi - lo) / np.where(val > 0, val, np.nan), np.nan)
        out[m] = {"auc_uncertain_removed": res_u, "auc_random_removed": res_r,
                  "u_lesion": float(np.nanmedian(u_full[ot & ok])), "u_contra": float(np.nanmedian(u_full[flip & ok]))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--name", help="results folder under out_root (default: the config name), e.g. full_aif__npde")
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    root = cfg.resolve("out_root") / (a.name or cfg.name)
    data_root = cfg.resolve("data_root")
    pids = [pid for pid in list_patients(root) if (root / subject_id(pid) / "uq" / "roi_posterior.csv").exists()]
    rng = np.random.default_rng(0)

    rows, sel, tau_items = [], {}, []
    for pid in pids:
        fd = root / subject_id(pid)
        with open(fd / "uq" / "roi_posterior.csv", encoding="utf8") as f:
            rr = list(csv.DictReader(f))
        rows += rr
        sel[pid] = selective(data_root / subject_id(pid), fd, rng)
        lab = nib.load(str(fd / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
        ot = nib.load(str(find_nii(data_root / subject_id(pid), "OT"))).get_fdata() > 0
        flip = np.flip(ot, axis=0) & ~ot
        z = np.load(fd / "uq" / "posterior.npz")
        for i, rid in enumerate(z["roi_ids"]):
            wl, wc = int((ot & (lab == rid)).sum()), int((flip & (lab == rid)).sum())
            if wl or wc:
                tau_items.append((wl, wc, z["tau"][i], float(np.exp(z["theta_map"][i][1]))))

    report = {"name": a.name or cfg.name, "n_subjects": len(pids), "n_rois": len(rows)}

    ident = {}
    for prm in PARAMS:
        if f"{prm}_post_sd" not in rows[0]:
            continue
        pinned = np.array([r[f"{prm}_pinned"] == "True" for r in rows])
        ratio = np.array([float(r[f"{prm}_post_sd"]) / max(float(r[f"{prm}_prior_sd"]), 1e-30) for r in rows])[~pinned]
        ident[prm] = {"pinned_frac": float(pinned.mean()),
                      "ratio_median": float(np.median(ratio)) if ratio.size else None,
                      "data_informed_frac": float(np.mean(ratio < 0.5)) if ratio.size else None,
                      "prior_dominated_frac": float(np.mean(ratio > 0.9)) if ratio.size else None}
    report["identifiability"] = ident

    size = defaultdict(dict)
    for q in ("CBF", "V", "MTT", "tau"):
        for tis in ("GM", "WM", "all"):
            rs = [r for r in rows if tis == "all" or r["tissue"] == tis]
            rel = [(float(r[f"{q}_hi"]) - float(r[f"{q}_lo"])) / max(float(r[f"{q}_mean"]), 1e-12) for r in rs]
            size[q][tis] = float(np.median(rel))
    report["rel_width_95"] = size
    report["convergence"] = {"rhat_gt_1_1_frac": float(np.mean([float(r["rhat_max"]) > 1.1 for r in rows])),
                             "ess_min_median": float(np.median([float(r["ess_min"]) for r in rows]))}
    report["fit_outside_ci_frac"] = {q: float(np.mean([not (float(r[f"{q}_lo"]) <= float(r[f"{q}_fit"]) <= float(r[f"{q}_hi"]))
                                                        for r in rows])) for q in ("CBF", "V", "MTT")}

    selrep = {}
    for m in ("CBF", "MTT"):
        au = np.array([s[m]["auc_uncertain_removed"] for s in sel.values() if m in s], dtype=float)
        ar = np.array([s[m]["auc_random_removed"] for s in sel.values() if m in s], dtype=float)
        ul = np.array([s[m]["u_lesion"] for s in sel.values() if m in s])
        uc = np.array([s[m]["u_contra"] for s in sel.values() if m in s])
        selrep[m] = {"fractions": FRACS, "auc_uncertain_removed": np.nanmean(au, 0).tolist(),
                     "auc_random_removed": np.nanmean(ar, 0).tolist(), "n": int(au.shape[0]),
                     "u_lesion_median": float(np.nanmedian(ul)), "u_contra_median": float(np.nanmedian(uc)),
                     "frac_patients_lesion_more_uncertain": float(np.nanmean(ul > uc))}
    report["selective"] = selrep

    if tau_items:
        wl = np.array([t[0] for t in tau_items], float)
        wc = np.array([t[1] for t in tau_items], float)
        draws = np.stack([t[2] for t in tau_items])
        tmap = np.array([t[3] for t in tau_items])
        diffs = []
        for j in range(draws.shape[1]):
            x = draws[:, j]
            diffs.append(wmedian(x[wl > 0], wl[wl > 0]) - wmedian(x[wc > 0], wc[wc > 0]))
        diffs = np.array(diffs)
        (root / "eval_uq").mkdir(exist_ok=True)
        np.save(root / "eval_uq" / "tau_diff_draws.npy", diffs)
        report["tau_lesion_minus_contra"] = {
            "at_fit": wmedian(tmap[wl > 0], wl[wl > 0]) - wmedian(tmap[wc > 0], wc[wc > 0]),
            "median": float(np.median(diffs)), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "prob_lesion_faster": float(np.mean(diffs < 0))}

    out = root / "eval_uq"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf8")

    print(f"[{a.name or cfg.name}] {len(pids)} subjects, {len(rows)} ROIs")
    print("  identifiability (posterior sd / prior sd; pinned = at a bound):")
    for prm, d in ident.items():
        rm = "–" if d["ratio_median"] is None else f"{d['ratio_median']:.2f}"
        di = "–" if d["data_informed_frac"] is None else f"{d['data_informed_frac']:.0%}"
        pd_ = "–" if d["prior_dominated_frac"] is None else f"{d['prior_dominated_frac']:.0%}"
        print(f"    {prm:8s} ratio {rm:>5}  learnt from data {di:>4}  prior-dominated {pd_:>4}  pinned {d['pinned_frac']:.0%}")
    print("  median relative 95% width (all / GM / WM): " + "  ".join(
        f"{q} {size[q]['all']:.0%}/{size[q]['GM']:.0%}/{size[q]['WM']:.0%}" for q in size))
    print(f"  convergence: {report['convergence']['rhat_gt_1_1_frac']:.1%} ROIs with R-hat > 1.1; "
          f"fitted map level outside the 95% CI: " + ", ".join(f"{q} {v:.0%}" for q, v in report["fit_outside_ci_frac"].items()))
    for m, d in selrep.items():
        print(f"  {m}: AUC removing most uncertain {', '.join(f'{x:.3f}' for x in d['auc_uncertain_removed'])} | random "
              f"{', '.join(f'{x:.3f}' for x in d['auc_random_removed'])}  (fractions {FRACS})")
        print(f"       relative 95% width lesion {d['u_lesion_median']:.0%} vs contralateral {d['u_contra_median']:.0%}; "
              f"lesion more uncertain in {d['frac_patients_lesion_more_uncertain']:.0%} of patients")
    if "tau_lesion_minus_contra" in report:
        d = report["tau_lesion_minus_contra"]
        print(f"  tau(lesion) - tau(contra): {d['at_fit']:+.1f} s at the fit; posterior median {d['median']:+.1f} s, "
              f"95% CI [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}] s, P(lesion faster) = {d['prob_lesion_faster']:.2f}")


if __name__ == "__main__":
    main()
