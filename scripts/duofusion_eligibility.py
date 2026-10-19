"""DuoFusion eligibility: conformal core volume x calibrated twin hypoperfusion.

python scripts/duofusion_eligibility.py --df24 <dir> --df18 <dir> --mv24 <dir> --mv18 <dir> --out <dir>

  df24, df18   scripts/duofusion_volumes.py outputs (per_patient.json, training_X.npz)
  mv24, mv18   scripts/multiverse_eligibility.py outputs (vendor-map volumes; ISLES 2018 truth)

1. Core (ISLES 2018, label = core on DWI right after the CTP), cross-validated over
   patients and repeated: residual conformal on log1p volumes around
     vendor  the vendor-map core (rCBF < 0.3, contralateral reference)
     twin    the DuoFusion point core at the rCBF threshold chosen on the training folds
   coverage and median width at 80% and 90%.
2. Hypoperfusion (ISLES'24, reference = vendor Tmax > 6 s volume, the trial definition):
   the twin point volume at the relative-Tmax threshold chosen on the training folds, with
     posterior  5-95% range of the twin posterior draws (no calibration)
     conformal  residual conformal on log1p volumes
3. Eligibility (ISLES'24): core draws from the ISLES 2018 residuals (both sources), hypo
   draws from the out-of-fold ISLES'24 hypo residuals, independent; the probabilities of
   DEFUSE 3 and DAWN, against the vendor-default decision and the multiverse share.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiverse_eligibility import dawn, defuse3  # noqa: E402

LEVELS = (0.8, 0.9)


def read_csv(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf8")):
        rec = {}
        for k, v in r.items():
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):
                rec[k] = None if v in ("", "None") else v
        out[int(rec["patient"])] = rec
    return out


def load_df(d):
    rows = {r["patient"]: r for r in json.load(open(Path(d) / "per_patient.json", encoding="utf8"))}
    for n, r in rows.items():
        z = np.load(Path(d) / f"training_{n}.npz")
        r["core_grid"], r["hypo_grid"] = z["core_grid"], z["hypo_grid"]
        r["tau_grid"], r["dt_grid"] = z["tau_grid"], z["dt_grid"]
    return rows


def cq(scores, alpha):
    n = len(scores)
    return float(np.sort(scores)[min(n, int(np.ceil((n + 1) * (1 - alpha)))) - 1])


def group_folds(groups, k, rng):
    keys = np.array(sorted(set(groups), key=str))
    rng.shuffle(keys)
    f = {g: i % k for i, g in enumerate(keys)}
    return np.array([f[g] for g in groups])


def residual_interval(e_train, base, alpha):
    return base - cq(-e_train, alpha / 2), base + cq(e_train, alpha / 2)


def choose_index(pred_by_idx, y):
    """Grid index whose log1p prediction has the smallest median absolute error."""
    return int(np.argmin([np.median(np.abs(y - p)) for p in pred_by_idx]))


def cv_conformal(y, preds, groups, repeats, seed=0):
    """preds: (n_grid, n) log1p predictions (one row for a fixed base). Returns coverage/width
    per level and the out-of-fold residuals of the last repeat."""
    res = {lv: {"cov": [], "wid": []} for lv in LEVELS}
    chosen = []
    oof = np.full(len(y), np.nan)
    for rep in range(repeats):
        fold = group_folds(groups, 5, np.random.default_rng(seed + rep))
        for k in range(5):
            te, tr = fold == k, fold != k
            j = choose_index(preds[:, tr], y[tr])
            chosen.append(j)
            e = y[tr] - preds[j, tr]
            if rep == repeats - 1:
                oof[te] = y[te] - preds[j, te]
            for lv in LEVELS:
                lo, hi = residual_interval(e, preds[j, te], 1 - lv)
                res[lv]["cov"] += list((lo <= y[te]) & (y[te] <= hi))
                res[lv]["wid"] += list(np.expm1(hi) - np.expm1(np.maximum(lo, 0)))
    out = {f"coverage_{int(100 * lv)}": float(np.mean(res[lv]["cov"])) for lv in LEVELS}
    out.update({f"width_ml_median_{int(100 * lv)}": float(np.median(res[lv]["wid"])) for lv in LEVELS})
    out["chosen_index_mode"] = int(np.bincount(chosen).argmax())
    return out, oof


def main():
    p = argparse.ArgumentParser()
    for k in ("df24", "df18", "mv24", "mv18", "out"):
        p.add_argument(f"--{k}", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--pairs", type=int, default=4000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df24, df18 = load_df(a.df24), load_df(a.df18)
    mv24, mv18 = read_csv(a.mv24 / "multiverse_per_patient.csv"), read_csv(a.mv18 / "multiverse_per_patient.csv")
    S = {}

    # 1. core on ISLES 2018
    n18 = sorted(n for n in df18 if n in mv18 and mv18[n].get("truth_core_ml") is not None)
    y18 = np.log1p([mv18[n]["truth_core_ml"] for n in n18])
    g18 = [mv18[n]["group"] for n in n18]
    tau_grid = df18[n18[0]]["tau_grid"]
    vend18 = np.log1p([[mv18[n]["core_default"] for n in n18]])
    twin18 = np.log1p(np.array([[df18[n]["core_grid"][0, i] for n in n18] for i in range(len(tau_grid))]))
    S["core_isles18"] = {"n_slabs": len(n18), "n_patients": len(set(g18))}
    S["core_isles18"]["vendor"], e_vend = cv_conformal(y18, vend18, g18, a.repeats)
    S["core_isles18"]["twin"], e_twin = cv_conformal(y18, twin18, g18, a.repeats)
    tau_star = float(tau_grid[S["core_isles18"]["twin"]["chosen_index_mode"]])
    S["core_isles18"]["twin"]["tau"] = tau_star
    e_core = {"vendor": y18 - vend18[0], "twin": y18 - twin18[list(tau_grid).index(tau_star)]}

    # 2. hypoperfusion on ISLES'24 against the vendor Tmax > 6 s volume
    n24 = sorted(n for n in df24 if n in mv24 and df24[n].get("vendor_hypo_ml") is not None)
    dt_grid = df24[n24[0]]["dt_grid"]
    yh = np.log1p([df24[n]["vendor_hypo_ml"] for n in n24])
    th = np.log1p(np.array([[df24[n]["hypo_grid"][0, j] for n in n24] for j in range(len(dt_grid))]))
    g24 = list(n24)
    S["hypo_isles24"], e_hypo_oof = cv_conformal(yh, th, g24, a.repeats)
    j_star = S["hypo_isles24"]["chosen_index_mode"]
    S["hypo_isles24"]["dt"] = float(dt_grid[j_star])
    post_cov = [np.percentile(df24[n]["hypo_grid"][1:, j_star], 5) <= df24[n]["vendor_hypo_ml"]
                <= np.percentile(df24[n]["hypo_grid"][1:, j_star], 95) for n in n24]
    post_w = [np.percentile(df24[n]["hypo_grid"][1:, j_star], 95) - np.percentile(df24[n]["hypo_grid"][1:, j_star], 5)
              for n in n24]
    S["hypo_isles24"]["posterior_5_95"] = {"coverage": float(np.mean(post_cov)), "width_ml_median": float(np.median(post_w))}
    dice = [df24[n]["dice_vs_vendor_by_dt"][j_star] for n in n24 if df24[n].get("dice_vs_vendor_by_dt")]
    S["hypo_isles24"]["dice_vs_vendor_median"] = float(np.median(dice))

    # 3. eligibility on ISLES'24
    rng = np.random.default_rng(1)
    rows = []
    i_tau = list(tau_grid).index(tau_star)
    for idx, n in enumerate(n24):
        m = mv24[n]
        e_h = np.delete(e_hypo_oof, idx)
        e_h = e_h[np.isfinite(e_h)]
        hyp = np.expm1(th[j_star, idx] + rng.choice(e_h, a.pairs))
        rec = {"patient": n, "age": m["age"], "nihss": m["nihss"], "vendor_core": m["core_default"],
               "vendor_hypo": df24[n]["vendor_hypo_ml"], "twin_hypo": float(np.expm1(th[j_star, idx])),
               "vendor_defuse3": m["defuse3_default"] == "True", "multiverse_defuse3": m["defuse3_share"],
               "vendor_dawn": m["dawn_default"], "multiverse_dawn": m["dawn_share"]}
        for src in ("vendor", "twin"):
            base = m["core_default"] if src == "vendor" else float(df24[n]["core_grid"][0, i_tau])
            core = np.expm1(np.log1p(base) + rng.choice(e_core[src], a.pairs))
            rec[f"{src}_core_point"] = base
            rec[f"p_defuse3_{src}"] = float(np.mean([defuse3(c, max(h, c)) for c, h in zip(core, hyp)]))
            dv = [dawn(c, m["age"], m["nihss"]) for c in core]
            rec[f"p_dawn_{src}"] = float(np.mean(dv)) if dv[0] is not None else None
        rows.append(rec)
    with open(a.out / "eligibility_isles24.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def bands(v):
        v = [x for x in v if x is not None]
        return {"n": len(v), "no(<=0.1)": int(sum(x <= 0.1 for x in v)), "uncertain": int(sum(0.1 < x < 0.9 for x in v)),
                "yes(>=0.9)": int(sum(x >= 0.9 for x in v))}

    def brier(ps, ref):
        pr = [(x, float(r)) for x, r in zip(ps, ref) if x is not None and r is not None and r != "None"]
        return float(np.mean([(x - r) ** 2 for x, r in pr])) if pr else None

    E = {"n": len(rows)}
    for crit in ("defuse3", "dawn"):
        ref = [r[f"vendor_{crit}"] in (True, "True") if r[f"vendor_{crit}"] is not None else None for r in rows]
        E[crit] = {"vendor_eligible": int(sum(bool(x) for x in ref if x is not None)),
                   "multiverse": bands([r[f"multiverse_{crit}"] for r in rows])}
        for src in ("vendor", "twin"):
            ps = [r[f"p_{crit}_{src}"] for r in rows]
            E[crit][src] = {**bands(ps), "brier_vs_vendor_decision": brier(ps, ref)}
    S["eligibility_isles24"] = E
    (a.out / "duofusion_summary.json").write_text(json.dumps(S, indent=1), encoding="utf8")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
