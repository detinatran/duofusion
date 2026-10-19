"""Fast cohort evaluation for any model variant.

Re-implements, with the same formulas, the two headline evaluations of
HemoPIC's Eval/ (MIT License, LDR Group @ JHU):
  * tracer reconstruction nRMSE per region (Eval/tracer_recon): the original
    script re-simulates curves with the original model from roi_fit_summary.csv,
    which is wrong for new model variants, so here the fitted ROI curves are read
    from fit_curves.npz instead;
  * lesion ROC AUC of CBF/CBV/MTT relative to the brain median (Eval/lesion_roc),
    without writing gigabytes of curve CSVs.
"""
import csv

import numpy as np

from .io import SEG4_REL, find_nii, load_nifti, squeeze_ctc, subject_id
from .signal import compute_peak_map, smooth_curve_gaussian, weight_from_aif

REGIONS = ("GM", "WM", "lesion", "c_lesion", "full_brain")
MAPS = (("CBF", "low"), ("CBV", "low"), ("MTT", "high"))
DECONV = {"CBF": "MR_rCBF", "CBV": "MR_rCBV", "MTT": "MR_MTT"}
SMOOTH_SIGMA, W_THR = 1.8, 0.30


def roc_auc(y_true, score):
    ok = np.isfinite(score)
    y, s = y_true[ok].astype(np.uint8), score[ok].astype(np.float64)
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    idx = np.r_[np.where(np.diff(ss) != 0.0)[0], ys.size - 1]
    tpr = np.r_[0.0, np.cumsum(ys == 1)[idx] / n_pos, 1.0]
    fpr = np.r_[0.0, np.cumsum(ys == 0)[idx] / n_neg, 1.0]
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) * 0.5))


def _lesion_scores(data, brain, direction):
    valid = brain & np.isfinite(data) & (data > 0.0)
    if not np.any(valid):
        return valid, None
    rel = data[valid] / float(np.median(data[valid]))
    score = 1.0 / np.maximum(rel, 1e-12) if direction == "low" else rel
    return valid, score


def _weighted_errors(c_obs, c_fit, w_t, w_mask):
    ww = w_mask.astype(np.float64) * w_t
    wsum = float(np.sum(ww))
    if wsum <= 0.0:
        return float("nan"), float("nan")
    diff2 = (c_fit - c_obs) ** 2
    rmse = float(np.sqrt(np.sum(ww * diff2) / wsum))
    peak = float(np.max(c_obs[w_mask])) if np.any(w_mask) else float(np.max(c_obs))
    nrmse = rmse / peak if np.isfinite(peak) and peak > 0 else float("nan")
    nmse = float(np.sum(ww * diff2) / (np.sum(ww * c_obs ** 2) + 1e-12))
    return nrmse, nmse


def evaluate_subject(data_root, fit_root, patient_num):
    base = data_root / subject_id(patient_num)
    fit_dir = fit_root / subject_id(patient_num)
    rows = []

    _, sub3 = load_nifti(fit_dir / "roi_labels_seg4_within_kmeans.nii.gz")
    sub3 = sub3.astype(np.int32)
    ctc = squeeze_ctc(load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))[1])
    info = __import__("json").loads((fit_dir / "run_info_core.json").read_text(encoding="utf8"))
    bf, thr = int(info["baseline_frames"]), float(info["aif_peak_threshold"])
    ot_path = find_nii(base, "OT")
    ot = (load_nifti(ot_path)[1] > 0) if ot_path is not None else np.zeros(sub3.shape, bool)

    show = (sub3 > 0) & np.all(np.isfinite(ctc), axis=3)
    fit_mask = show & (compute_peak_map(ctc, bf) > thr)
    if not np.any(fit_mask):
        fit_mask = show
    cart = smooth_curve_gaussian(ctc[fit_mask].mean(axis=0), SMOOTH_SIGMA, bf, True)
    cart = cart / max(float(np.max(cart)), 1e-12)
    w_t = weight_from_aif(cart)
    w_mask = w_t >= W_THR
    if int(np.sum(w_mask)) < 3:
        w_mask = w_t >= 0.10

    tissue = {}
    with open(fit_dir / "roi_fit_summary.csv", encoding="utf8") as f:
        for r in csv.DictReader(f):
            tissue[int(r["roi_id"])] = r["tissue"]
    z = np.load(fit_dir / "fit_curves.npz")
    curves = {int(rid): z["c_fit"][i] for i, rid in enumerate(z["roi_ids"]) if bool(z["success"][i])}

    gm = np.isin(sub3, [k for k, v in tissue.items() if v == "GM"]) & fit_mask
    wm = np.isin(sub3, [k for k, v in tissue.items() if v == "WM"]) & fit_mask
    masks = {"GM": gm, "WM": wm, "lesion": ot & fit_mask, "c_lesion": np.flip(ot, axis=0) & fit_mask, "full_brain": fit_mask}
    for reg in REGIONS:
        m = masks[reg]
        nrmse = nmse = float("nan")
        if np.any(m) and curves:
            c_obs = smooth_curve_gaussian(ctc[m].mean(axis=0), SMOOTH_SIGMA, bf, True)
            num, den = np.zeros(ctc.shape[3]), 0.0
            for rid, cf in curves.items():
                a = int(np.sum(m & (sub3 == rid)))
                if a:
                    num += a * cf
                    den += a
            if den > 0:
                nrmse, nmse = _weighted_errors(c_obs, num / den, w_t, w_mask)
        rows.append({"patient": patient_num, "kind": "tracer", "region": reg, "method": "model", "value": nrmse, "nmse": nmse})

    seg_path = base / SEG4_REL
    brain = load_nifti(seg_path)[1] > 0
    pooled = {}
    for name, direction in MAPS:
        for method, path in (("model", fit_dir / f"HemoPIC_{name}.nii.gz"), ("deconv", find_nii(base, DECONV[name]))):
            data = load_nifti(path)[1].astype(np.float64)
            valid, score = _lesion_scores(data, brain, direction)
            auc = roc_auc(ot[valid], score) if score is not None else float("nan")
            rows.append({"patient": patient_num, "kind": "roc", "region": name, "method": method, "value": auc, "nmse": ""})
            if score is not None:
                pooled[(method, name)] = (ot[valid].astype(np.uint8), score.astype(np.float64))
    return rows, pooled
