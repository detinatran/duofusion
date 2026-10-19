"""DuoFusion: joint posterior of the core and hypoperfused volumes from the HemoPIC+ twin.

python scripts/duofusion_volumes.py <data_root> <twin_out> <out_dir> [--draws 100] [--workers 8]

Needs, per patient, the HemoPIC+ fits of external/HemoPIC_plus (scripts/run_twin.sh):
  <twin_out>/isles24_full_aif/training_X/            ROI fits, ROI labels
  <twin_out>/isles24_full_aif/training_X/uq/posterior.npz   ROI posterior draws (run_uq.py)
  <twin_out>/isles24_full_aif_idvox_osvd_nc/training_X/HemoPIC_CBV.nii.gz   indicator-dilution CBV

For every tissue voxel, the voxel fit x = (log V, log Cin, delay) of npde/voxel.py, its
tempered Laplace covariance (residual dof 3) and its linear response dx/dtheta to the ROI
parameters (implicit function theorem, as run_voxel_uq.py). Draw k combines one ROI
posterior draw theta_k (shared by the voxels of the ROI) with independent voxel noise:
  log MTT_k = log MTT + (dx_0/dtheta - dlog CBF_roi/dtheta) (theta_k - theta) + e_0
  delay_k   = delay   +  dx_2/dtheta (theta_k - theta) + e_2,      e ~ N(0, Laplace cov)
  CBF_k     = 60 CBV_ID / MTT_k
  Tmax_k    = delay_k + t*(beta_k, MTT_k),  t* = beta MTT ln(MTT/beta) / (MTT - beta)
where t* is the time to peak of the residue of the compartment (mean transit time MTT)
fed through the exponential input dispersion of mean beta (the ROI's dispersion): the
model's counterpart of the deconvolution Tmax. Per draw (k = 0: the fit itself):
  core = rCBF < tau (tau 0.2, 0.3) with the contralateral healthy (0 <= Tmax < 4 s) median,
  hypo = Tmax > 6 s or core.
Every draw is smoothed (nan-aware Gaussian, --sigma mm, default 3) before thresholding,
and Tmax is taken relative to the less delayed hemisphere (Tmax_rel = Tmax - its median),
since the twin's delay is measured from its own AIF and is offset from the vendor Tmax.
Volumes are stored on grids of thresholds (rCBF tau in TAU_GRID, Tmax_rel in DT_GRID,
and the union core | hypo for every pair) so that thresholds can be calibrated afterwards.
Writes <out_dir>/training_X.npz (draws), per_patient.csv (point values, draw quantiles,
agreement of Tmax_twin > 6 with the vendor Tmax > 6, core Dice/AUC against OT) and the
maps DF_CBF.nii.gz, DF_Tmax.nii.gz, DF_Tmax_sd.nii.gz, DF_core_prob.nii.gz per patient.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "HemoPIC_plus"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from duofusion.core import dice, roc_auc  # noqa: E402
from duofusion.data import load_subject  # noqa: E402
from duofusion.reference import affected_side, nan_gaussian, spacing_of  # noqa: E402
from hemopic_plus.npde.voxel import RoiModel, fit_voxels, roi_cbf, roi_sensitivity, smooth_curves, voxel_laplace  # noqa: E402
from hemopic_plus.uq.data import load_subject as load_fit  # noqa: E402

torch.set_num_threads(1)
NAMES7 = ["log_R", "log_tau", "log_V", "log_F", "log_Cin", "delay", "log_disp"]
TAUS = (0.2, 0.3)
HEALTHY, HYPO = 4.0, 6.0                 # vendor Tmax (s)
REL_HEALTHY, REL_HYPO = 2.0, 4.0         # twin Tmax relative to the less delayed hemisphere (s), point summaries
TAU_GRID = np.round(np.arange(0.10, 0.551, 0.05), 3)
DT_GRID = np.round(np.arange(0.0, 8.01, 0.5), 2)


def residue_peak(beta, mtt):
    """Time to peak of exp(-t/MTT) convolved with the dispersion kernel exp(-t/beta)/beta."""
    beta = np.maximum(beta, 1e-3)
    mtt = np.maximum(mtt, 1e-3)
    close = np.abs(mtt - beta) < 1e-3 * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta * mtt * np.log(mtt / beta) / (mtt - beta)
    return np.where(close, beta, t)


def roi_terms(fit, ctc, lab, brain, post, rid, i):
    """Voxel fit, Laplace covariance and ROI-parameter response for the voxels of one ROI."""
    rcfg = fit.cfg
    t, dt, shape, w = fit.t.numpy(), fit.dt, fit.shape.numpy(), fit.w[0].numpy()
    th = np.r_[fit.theta_fit[i].numpy(), 0.0]

    def make(th_):
        return RoiModel(rcfg, t, dt, shape, th_, n_cells=1)
    model = make(th)
    vox = np.argwhere(brain & (lab == rid))
    c = smooth_curves(ctc[tuple(vox.T)], rcfg.smooth_sigma_frames, rcfg.baseline_frames, rcfg.clip_nonneg)
    x, _, _ = fit_voxels(model, c, w)
    cov, _, _ = voxel_laplace(model, x, c, w, sigma_s=rcfg.smooth_sigma_frames, dof=3)
    cov = np.asarray(cov)
    dx, dl = roi_sensitivity(make, th, x, c, w, rcfg.tail_frac)
    dx, dl = np.asarray(dx)[:, :, :7], np.asarray(dl)[:7]
    names = [str(s) for s in post["names"]]
    assert names == NAMES7, names
    draws = post["theta"][list(post["roi_ids"]).index(rid)]           # (S, 7)
    vals, vecs = np.linalg.eigh(cov)
    chol = vecs * np.sqrt(np.clip(vals, 0.0, None))[:, None, :]
    return {"vox": vox, "x": np.asarray(x), "cbf_roi": roi_cbf(model, rcfg.tail_frac),
            "g_mtt": dx[:, 0, :] - dl[None, :], "g_delay": dx[:, 2, :], "chol": chol,
            "theta": th[:7], "draws": draws}


def work(args):
    data_root, duo, out, num, n_draws, seed, sigma, roi_scale = args
    t0 = time.time()
    fit_dir = duo / "isles24_full_aif" / f"training_{num}"
    post_f = fit_dir / "uq" / "posterior.npz"
    cbv_f = duo / "isles24_full_aif_idvox_osvd_nc" / f"training_{num}" / "HemoPIC_CBV.nii.gz"
    if not (post_f.exists() and cbv_f.exists()):
        return {"patient": num, "skipped": "missing fits"}
    if (out / f"training_{num}.npz").exists():
        return {"patient": num, "skipped": "done"}
    s = load_subject(data_root, num)
    fit = load_fit(fit_dir)
    ctc = s.ctc()
    lab = np.rint(nib.load(str(fit_dir / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata()).astype(int)
    cbv_id = nib.load(str(cbv_f)).get_fdata()
    post = np.load(post_f)
    brain = (s.tissue > 0) & np.all(np.isfinite(ctc), axis=3) & (lab > 0) & np.isfinite(cbv_id) & (cbv_id > 0)
    terms = [roi_terms(fit, ctc, lab, brain, post, int(rid), i) for i, rid in enumerate(fit.roi_ids)
             if np.any(brain & (lab == int(rid)))]
    rng = np.random.default_rng(seed + num)
    hemi, tissue, ot = s.hemi, s.tissue, s.ot
    vendor_tmax = s.maps.get("Tmax")

    def draw(k):
        logmtt = np.full(lab.shape, np.nan)
        delay = np.full(lab.shape, np.nan)
        beta = np.full(lab.shape, np.nan)
        for T in terms:
            idx = tuple(T["vox"].T)
            if k == 0:
                dth, e = np.zeros(7), np.zeros((len(T["x"]), 3))
                b = np.exp(T["theta"][6])
            else:
                d = T["draws"][rng.integers(len(T["draws"]))]
                dth = roi_scale * (d - T["draws"].mean(axis=0))   # posterior spread around the fit (linearised, as run_voxel_uq.py)
                e = np.einsum("nij,nj->ni", T["chol"], rng.standard_normal((len(T["x"]), 3)))
                b = np.exp(T["theta"][6] + dth[6])
            logmtt[idx] = np.log(60.0) + T["x"][:, 0] - np.log(T["cbf_roi"]) + T["g_mtt"] @ dth + e[:, 0]
            delay[idx] = T["x"][:, 2] + T["g_delay"] @ dth + e[:, 2]
            beta[idx] = b
        mtt = np.exp(logmtt)
        cbf = 60.0 * cbv_id / mtt
        tmax = delay + residue_peak(beta, mtt)
        return cbf, tmax

    sp = spacing_of(s.affine)

    def smooth(vol, m):
        return nan_gaussian(np.where(m, vol, np.nan), sigma, sp) if sigma > 0 else np.where(m, vol, np.nan)

    rows_k, keep = [], {}
    grids = {"core_grid": [], "hypo_grid": [], "union_grid": []}
    probs = {tau: np.zeros(lab.shape) for tau in TAUS}
    tmax_sum, tmax_sq = np.zeros(lab.shape), np.zeros(lab.shape)
    for k in range(n_draws + 1):
        cbf, tmax = draw(k)
        raw_ok = brain & np.isfinite(cbf) & (cbf > 0) & np.isfinite(tmax)
        cbf, tmax = smooth(cbf, raw_ok), smooth(tmax, raw_ok)
        valid = brain & np.isfinite(cbf) & (cbf > 0) & np.isfinite(tmax)
        med = [np.median(tmax[valid & (hemi == h)]) if np.any(valid & (hemi == h)) else np.inf for h in (1, 2)]
        base = min(med) if np.isfinite(min(med)) else np.median(tmax[valid])
        trel = tmax - base
        healthy = valid & (trel < REL_HEALTHY)
        hypo = valid & (trel > REL_HYPO)
        side = affected_side(s, hypo)
        contra = (hemi == 3 - side) if side else (hemi > 0)
        sel = healthy & contra if np.any(healthy & contra) else healthy
        ref = float(np.median(cbf[sel])) if np.any(sel) else float(np.median(cbf[valid]))
        r = cbf / ref
        cores = np.stack([valid & (r < t_) for t_ in TAU_GRID])          # (nT, ...)
        hypos = np.stack([valid & (trel > d_) for d_ in DT_GRID])        # (nD, ...)
        grids["core_grid"].append(cores.reshape(len(TAU_GRID), -1).sum(1) * s.voxel_ml)
        grids["hypo_grid"].append(hypos.reshape(len(DT_GRID), -1).sum(1) * s.voxel_ml)
        grids["union_grid"].append(np.array([[float((c_ | h_).sum()) for h_ in hypos] for c_ in cores]) * s.voxel_ml)
        rec = {"hypo_ml": float((hypo | (valid & (r < max(TAUS)))).sum()) * s.voxel_ml}
        for tau in TAUS:
            core = valid & (r < tau)
            rec[f"core{int(100 * tau)}_ml"] = float(core.sum()) * s.voxel_ml
            rec[f"hypo{int(100 * tau)}_ml"] = float((hypo | core).sum()) * s.voxel_ml
            if k:
                probs[tau] += core
        rows_k.append(rec)
        if k == 0:
            keep = {"cbf": cbf, "tmax": tmax, "trel": trel, "valid": valid, "r": r, "hypo": hypo}
        else:
            t_ = np.where(valid, tmax, 0.0)
            tmax_sum += t_
            tmax_sq += t_ ** 2
    od = out / f"training_{num}"
    od.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(fit_dir / "roi_labels_seg4_within_kmeans.nii.gz"))
    sd = np.sqrt(np.maximum(tmax_sq / n_draws - (tmax_sum / n_draws) ** 2, 0.0))
    for name, vol in (("DF_CBF", keep["cbf"]), ("DF_Tmax", keep["tmax"]), ("DF_Tmax_rel", keep["trel"]), ("DF_Tmax_sd", np.where(keep["valid"], sd, np.nan)),
                      ("DF_core_prob", np.where(keep["valid"], probs[0.3] / n_draws, np.nan))):
        nib.save(nib.Nifti1Image(np.asarray(vol, np.float32), img.affine), str(od / f"{name}.nii.gz"))
    arr = {key: np.array([r[key] for r in rows_k]) for key in rows_k[0]}
    np.savez_compressed(out / f"training_{num}.npz", tau_grid=TAU_GRID, dt_grid=DT_GRID,
                        **{k: np.array(v) for k, v in grids.items()}, **arr)
    v, y = keep["valid"], ot & keep["valid"]
    res = {"patient": num, "seconds": round(time.time() - t0, 1), "truth_ml": float(y.sum()) * s.voxel_ml,
           "brain_ml": float(v.sum()) * s.voxel_ml}
    for key, a in arr.items():
        res[f"{key}_point"] = float(a[0])
        res[f"{key}_p10"], res[f"{key}_p50"], res[f"{key}_p90"] = (float(q) for q in np.percentile(a[1:], [10, 50, 90]))
    if vendor_tmax is not None:
        vt = v & np.isfinite(vendor_tmax) & (vendor_tmax >= 0)
        vs = nan_gaussian(np.where(vt, vendor_tmax, np.nan), sigma, sp) if sigma > 0 else vendor_tmax
        ok = vt & np.isfinite(vs)
        res["tmax_rel_corr_vs_vendor"] = float(np.corrcoef(keep["trel"][ok], vs[ok])[0, 1]) if ok.sum() > 10 else np.nan
        vh = ok & (vendor_tmax > HYPO)
        res["vendor_hypo_ml"] = float(vh.sum()) * s.voxel_ml
        res["dice_vs_vendor_by_dt"] = [dice((ok & (keep["trel"] > d_))[ok], vh[ok]) for d_ in DT_GRID]
        res["vol_by_dt"] = [float((ok & (keep["trel"] > d_)).sum()) * s.voxel_ml for d_ in DT_GRID]
    if y.any():
        res["core_auc"] = roc_auc(y[v], -np.log(keep["r"][v]))
        res["core30_dice"] = dice((v & (keep["r"] < 0.3))[v], y[v])
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=Path)
    p.add_argument("twin_out", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--draws", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian smoothing (mm) of every draw")
    p.add_argument("--roi-scale", type=float, default=1.0, help="scale of the ROI posterior part (0 = voxel noise only)")
    a = p.parse_args()
    nums = a.patients or sorted(int(d.name.split("_")[1]) for d in (a.twin_out / "isles24_full_aif").glob("training_*"))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, [(a.data_root, a.twin_out, a.out_dir, n, a.draws, a.seed, a.sigma, a.roi_scale) for n in nums]):
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                              if not isinstance(v, list)}), flush=True)
            if "skipped" not in r:
                rows.append(r)
    if rows:
        with open(a.out_dir / "per_patient.json", "w", encoding="utf8") as f:
            json.dump(rows, f)
        rows = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "patient", k))
        with open(a.out_dir / "per_patient.csv", "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
