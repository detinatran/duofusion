"""Lesion metrics beyond the ROC AUC, shared by the ISLES 2017 and ISLES'24 scripts.

Every map is scored relative to its brain median, as the AUC of evaluation.py:
  * average precision (area under the precision-recall curve), which, unlike
    the AUC, depends on the small lesion fraction;
  * Dice and volume of the thresholded map, on a grid of relative thresholds
    (CBF and CBV below, MTT above). Two thresholds are reported: a fixed one from
    the CT perfusion literature (relative CBF < 30% for the infarct core, relative
    MTT > 145% for the hypoperfused tissue) and one chosen by leave-one-patient-out
    cross-validation (the threshold with the best mean Dice on the other
    patients, so no patient is scored with a threshold tuned on it);
  * patient-level bootstrap 95% intervals of every cohort mean and of paired
    differences.
"""
import numpy as np

GRID = {"low": np.round(np.arange(0.05, 1.0001, 0.025), 4), "high": np.round(np.arange(1.05, 6.0001, 0.05), 4)}
FIXED = {"CBF": 0.30, "MTT": 1.45}


def average_precision(y, score):
    ok = np.isfinite(score)
    y, s = y[ok].astype(bool), score[ok]
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    last = np.r_[np.where(np.diff(ss) != 0.0)[0], ys.size - 1]
    tp = np.cumsum(ys)[last]
    fp = (last + 1) - tp
    prec, rec = tp / (tp + fp), tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def map_metrics(vol, brain, ot, direction, voxel_ml):
    """AUC is computed by the caller; here AP, and Dice / predicted volume on GRID."""
    if vol is None:
        return None
    valid = brain & np.isfinite(vol) & (vol > 0.0)
    if not valid.any():
        return None
    rel = vol[valid] / float(np.median(vol[valid]))
    y = ot[valid].astype(bool)
    score = 1.0 / np.maximum(rel, 1e-12) if direction == "low" else rel
    grid = GRID[direction]
    pred = (rel[None, :] < grid[:, None]) if direction == "low" else (rel[None, :] > grid[:, None])
    inter = (pred & y[None, :]).sum(1)
    n_pred = pred.sum(1)
    dice = 2.0 * inter / np.maximum(n_pred + y.sum(), 1)
    return {"ap": average_precision(y, score), "dice": dice.astype(float), "vol_pred": (n_pred * voxel_ml).astype(float),
            "vol_true": float(y.sum() * voxel_ml), "prevalence": float(y.mean())}


def cv_threshold(dice_by_patient):
    """Leave-one-patient-out: index of the threshold with the best mean Dice on the others."""
    D = np.asarray(dice_by_patient)
    tot = np.nansum(D, 0)
    cnt = np.sum(np.isfinite(D), 0)
    idx = []
    for i in range(D.shape[0]):
        others = (tot - np.nan_to_num(D[i])) / np.maximum(cnt - np.isfinite(D[i]), 1)
        idx.append(int(np.nanargmax(others)))
    return np.array(idx)


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, (n, x.size))].mean(1)
    return [float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(records, methods, maps, auc_key="auc"):
    """records: per patient {method: {map: {"auc", "ap", "dice", "vol_pred", "vol_true"}}}.
    Returns per-patient rows and the cohort summary."""
    from scipy.stats import spearmanr
    rows = [{"patient": r["patient"]} for r in records]
    summ = {"n_patients": len(records), "methods": {}}
    for meth in methods:
        sm = {}
        for m, direction in maps:
            cell = [r.get(meth, {}).get(m) for r in records]
            ok = [c is not None for c in cell]
            if not any(ok):
                continue
            auc = np.array([c[auc_key] if c else np.nan for c in cell])
            ap = np.array([c["ap"] if c else np.nan for c in cell])
            grid = GRID[direction]
            D = np.array([c["dice"] if c else np.full(grid.size, np.nan) for c in cell])
            VP = np.array([c["vol_pred"] if c else np.full(grid.size, np.nan) for c in cell])
            vt = np.array([c["vol_true"] if c else np.nan for c in cell])
            j = cv_threshold(D)
            dice_cv = D[np.arange(len(cell)), j]
            vol_cv = VP[np.arange(len(cell)), j]
            out = {"AUC": bootstrap_ci(auc), "AP": bootstrap_ci(ap), "Dice (CV threshold)": bootstrap_ci(dice_cv),
                   "CV threshold, median": float(np.median(grid[j])),
                   "volume abs error ml (CV threshold)": bootstrap_ci(np.abs(vol_cv - vt)),
                   "volume Spearman (CV threshold)": float(spearmanr(vol_cv, vt, nan_policy="omit")[0])}
            if m in FIXED:
                k = int(np.argmin(np.abs(grid - FIXED[m])))
                out[f"Dice (fixed {FIXED[m]:g})"] = bootstrap_ci(D[:, k])
                out[f"volume Spearman (fixed {FIXED[m]:g})"] = float(spearmanr(VP[:, k], vt, nan_policy="omit")[0])
            sm[m] = out
            for i, row in enumerate(rows):
                row[f"{meth} {m} AUC"] = auc[i]
                row[f"{meth} {m} AP"] = ap[i]
                row[f"{meth} {m} Dice CV"] = dice_cv[i]
                row[f"{meth} {m} vol pred CV"] = vol_cv[i]
                row["lesion volume ml"] = vt[i] if np.isfinite(vt[i]) else row.get("lesion volume ml", np.nan)
        if sm:
            sm["mean AUC"] = bootstrap_ci([np.nanmean([rows[i].get(f"{meth} {m} AUC", np.nan) for m, _ in maps])
                                           for i in range(len(rows))])
            sm["mean AP"] = bootstrap_ci([np.nanmean([rows[i].get(f"{meth} {m} AP", np.nan) for m, _ in maps])
                                          for i in range(len(rows))])
            summ["methods"][meth] = sm
    return rows, summ


def paired_bootstrap(rows, a, b, n=2000, seed=0):
    d = np.array([r.get(a, np.nan) - r.get(b, np.nan) for r in rows], float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    from scipy.stats import wilcoxon
    ci = bootstrap_ci(d, n, seed)
    return {"diff": ci[0], "ci": ci[1:], "p_wilcoxon": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, "n": int(d.size)}
