"""Core delineation from relative CBF and its evaluation against an infarct mask."""
import numpy as np
from scipy.stats import rankdata

THRESHOLDS = (0.2, 0.3)


def roc_auc(y, score):
    """Rank-based ROC AUC (ties averaged); NaN if a class is missing."""
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(y, score):
    y = np.asarray(y, bool)
    if not np.any(y):
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="stable")
    hits = np.cumsum(y[order])
    prec = hits / np.arange(1, y.size + 1)
    return float(np.sum(prec[y[order]]) / y.sum())


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2 * int((a & b).sum()) / s) if s else float("nan")


def evaluate(cbf, ref, truth, region, voxel_ml, hypo=None, thresholds=THRESHOLDS):
    """Metrics of the relative CBF cbf / ref inside region (bool) against truth (bool).

    AUC/AP use -log(rCBF) over every region voxel with a finite ratio. The core at a
    threshold is rCBF < tau; with hypo given, also the Tmax-restricted core (rCBF < tau
    inside the hypoperfused tissue). Volumes in ml, signed error = core - truth."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cbf / ref
    ok = region & np.isfinite(r) & (r > 0)
    y = truth & ok
    out = {"n_vox": int(ok.sum()), "truth_ml": float(y.sum()) * voxel_ml}
    if not np.any(ok):
        return out
    score = -np.log(r[ok])
    out["auc"] = roc_auc(y[ok], score)
    out["ap"] = average_precision(y[ok], score)
    for tau in thresholds:
        variants = {"": ok & (r < tau)}
        if hypo is not None:
            variants["_tmax"] = ok & (r < tau) & hypo
        for suffix, core in variants.items():
            key = f"{int(round(100 * tau))}{suffix}"
            out[f"dice{key}"] = dice(core, y)
            out[f"core_ml{key}"] = float(core.sum()) * voxel_ml
            out[f"volerr_ml{key}"] = out[f"core_ml{key}"] - out["truth_ml"]
    return out


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """Mean of a - b over patients (pairs with both values finite) and its percentile 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if d.size == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": int(d.size), "mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "wins": int(np.sum(d > 0))}
