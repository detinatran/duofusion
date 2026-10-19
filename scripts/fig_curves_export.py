"""Real curves of one ISLES'24 voxel for the pipeline figure of the truncation study.

python scripts/fig_curves_export.py <data_root> <twin_full> <twin_trunc> <out.npz> --seconds 30
       [--patients 1 2 ...] [--models $DATA_DIR/learned_extrap] [--folds scripts/learned_folds24.json]

Picks a clean, high-amplitude, late-washout grey-matter voxel with a delayed vendor Tmax
(6-12 s, hypoperfused tissue) from a long scan, truncates its curve (and the AIF) to the first L seconds exactly
as truncate_ctp.py does, and runs every estimator of the study on it with the code of
osvd_volumes.py / duofusion_volumes.py:
  osvd, ssvd     residue k of the truncated curve; fit = AIF (*) k on the measured frames
  gamma          gamma_extend of the truncated curve to 70 s
  learned        learned.extend with the model of the patient's held-out fold
  twin           HemoPIC+ voxel fit (fit_voxels, n_cells = 1) on the truncated scan, its ROI
                 from the truncated HemoPIC+ fit; the fitted model is then simulated over the
                 full time axis, its AIF (the truncated fit's own) extended by gamma_extend
                 (only information measured in the first L seconds)
plus, on the full scan, the sSVD residue (deconvolution panel) and the full HemoPIC+ voxel fit
with its Windkessel outflow q(t) (twin panel). All curves in the units of the CTC (HU).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "HemoPIC_plus"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from duofusion import learned  # noqa: E402
from duofusion.data import load_subject  # noqa: E402
from duofusion.deconv import gamma_extend  # noqa: E402
from hemopic_plus.npde.voxel import RoiModel, fit_voxels, smooth_curves  # noqa: E402
from hemopic_plus.osvd import _circulant_svd, _oscillation  # noqa: E402
from hemopic_plus.signal import positive_excursion  # noqa: E402
from hemopic_plus.uq.data import load_subject as load_fit  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE  # noqa: E402

LAB = "roi_labels_seg4_within_kmeans.nii.gz"


def osvd_residue(c, aif, dt, osc_target=0.095,
                 thresholds=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)):
    """oSVD of osvd_maps for one curve: residue k (T,) and the fit AIF (*) k (T,)."""
    T = c.size
    U, S, Vt = _circulant_svd(aif, dt)
    cpad = np.r_[c, np.zeros(T)]
    for thr in thresholds:
        inv = np.where(S >= thr * S.max(), 1.0 / S, 0.0)
        k = Vt.T @ (inv * (U.T @ cpad))
        if _oscillation(k[None, :])[0] <= osc_target:
            break
    D = (U * S) @ Vt
    return k[:T], (D @ k)[:T]


def ssvd_residue(c, aif, dt, threshold=0.2):
    """sSVD of deconv.ssvd_maps for one curve: residue k (T,) and the fit AIF (*) k (T,)."""
    T = aif.size
    A = np.zeros((T, T))
    for i in range(T):
        A[i, : i + 1] = aif[i::-1] * dt
    U, S, Vt = np.linalg.svd(A)
    inv = np.where(S >= threshold * S.max(), 1.0 / S, 0.0)
    k = (Vt.T * inv) @ U.T @ c
    return k, A @ k


def twin_voxel(fit, lab, v, curve, t_sim=None, shape_sim=None):
    """HemoPIC+ voxel fit of one curve in its ROI; returns x, the fitted curve on the fit's own
    axis and, when t_sim/shape_sim are given, the model (C, q_out) simulated on that axis."""
    rid = int(lab[v])
    i = [int(r) for r in fit.roi_ids].index(rid)
    th = np.r_[fit.theta_fit[i].numpy(), 0.0]
    cfg = fit.cfg
    model = RoiModel(cfg, fit.t.numpy(), fit.dt, fit.shape.numpy(), th, n_cells=1)
    c = smooth_curves(curve[None, :], cfg.smooth_sigma_frames, cfg.baseline_frames, cfg.clip_nonneg)
    x, _, nmse = fit_voxels(model, c, fit.w[0].numpy())
    xt = torch.as_tensor(x, dtype=DTYPE)
    with torch.no_grad():
        q_fit, c_fit = model.simulate(xt)
    out = {"x": x[0], "nmse": float(nmse[0]), "smooth": c[0], "fit": c_fit[0].numpy(), "q": q_fit[0].numpy(),
           "theta": th}
    if t_sim is not None:
        sim = RoiModel(cfg, t_sim, fit.dt, shape_sim, th, n_cells=1)
        with torch.no_grad():
            q_s, c_s = sim.simulate(xt)
        out["sim"], out["q_sim"] = c_s[0].numpy(), q_s[0].numpy()
    return out


def candidates(s, ctc, lab_full, lab_tr, bf):
    """Score late-washout, low-noise voxels in the infarct with a delayed vendor Tmax."""
    tmax = s.maps.get("Tmax")
    m = (s.tissue == 1) & (lab_full > 0) & (lab_tr > 0) & np.all(np.isfinite(ctc), axis=3)
    if tmax is not None:
        m &= np.isfinite(tmax) & (tmax >= 6) & (tmax <= 12)
    vox = np.argwhere(m)
    if len(vox) == 0:
        return None
    y = positive_excursion(ctc[tuple(vox.T)], bf)
    ys = smooth_curves(y, 1.5, bf, True)
    tp = ys.argmax(1)
    pk = ys.max(1)
    noise = (y - ys).std(1) + 1e-6
    T = y.shape[1]
    tail = ys[:, -5:].mean(1) / np.maximum(pk, 1e-6)
    ok = (tp >= 18) & (tp <= 27) & (tail < 0.45) & (pk > np.percentile(pk, 90))
    if not np.any(ok):
        return None
    score = np.where(ok, pk / noise, -np.inf)
    j = int(score.argmax())
    return {"v": tuple(vox[j]), "score": float(score[j]), "tp": int(tp[j]), "peak": float(pk[j]), "T": T}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=Path)
    p.add_argument("twin_full", type=Path)
    p.add_argument("twin_trunc", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--patients", type=int, nargs="*", default=None)
    p.add_argument("--models", type=Path, default=Path.home() / "duofusion_data" / "learned_extrap")
    p.add_argument("--folds", type=Path, default=ROOT / "scripts" / "learned_folds24.json")
    a = p.parse_args()
    folds = {int(k): v for k, v in json.loads(a.folds.read_text()).items()}
    full_root, tr_root = a.twin_full / "isles24_full_aif", a.twin_trunc / "isles24_full_aif"
    nums = a.patients or sorted(int(d.name.split("_")[1]) for d in tr_root.glob("training_*"))

    best = None
    for n in nums:
        fd, td = full_root / f"training_{n}", tr_root / f"training_{n}"
        if not ((fd / "fit_curves.npz").exists() and (td / "fit_curves.npz").exists()):
            continue
        s = load_subject(a.data_root, n)
        ctc = s.ctc()
        if ctc.shape[3] < 60:
            continue
        lab_f = np.rint(nib.load(str(fd / LAB)).get_fdata()).astype(int)
        lab_t = np.rint(nib.load(str(td / LAB)).get_fdata()).astype(int)
        cfg = load_fit(fd).cfg
        c = candidates(s, ctc, lab_f, lab_t, cfg.baseline_frames)
        if c and (best is None or c["score"] > best["score"]):
            best = {**c, "patient": n}
            print("candidate", best, flush=True)
    if best is None:
        sys.exit("no voxel found")

    n, v = best["patient"], best["v"]
    fd, td = full_root / f"training_{n}", tr_root / f"training_{n}"
    s = load_subject(a.data_root, n)
    ctc = s.ctc()
    fit_f, fit_t = load_fit(fd), load_fit(td)
    lab_f = np.rint(nib.load(str(fd / LAB)).get_fdata()).astype(int)
    lab_t = np.rint(nib.load(str(td / LAB)).get_fdata()).astype(int)
    cfg = fit_f.cfg
    bf = cfg.baseline_frames
    dt = float(fit_f.dt)
    keep = int(np.floor(a.seconds / dt)) + 1               # as truncate_ctp.py
    raw_full = ctc[v].astype(np.float64)
    raw_tr = raw_full[:keep]
    t_full = np.arange(raw_full.size) * dt
    t70 = np.arange(learned.TMAX) * dt

    aif_full = positive_excursion(np.load(fd / "fit_curves.npz")["cart_norm"][None, :], bf)[0]
    aif_tr_raw = np.load(td / "fit_curves.npz")["cart_norm"]
    aif_tr = positive_excursion(aif_tr_raw[None, :], bf)[0]
    truth = positive_excursion(raw_full[None, :], bf)[0]
    obs = positive_excursion(raw_tr[None, :], bf)[0]

    k_o, fit_o = osvd_residue(obs, aif_tr, dt)
    k_s, fit_s = ssvd_residue(obs, aif_tr, dt)
    gam = gamma_extend(obs[None, :], dt, t70[-1])[0]
    ms = learned.load_models(a.models, [folds[n]] if n in folds else None, "cpu")
    lrn = positive_excursion(learned.extend(raw_tr[None, :], ms, "cpu").astype(np.float64), bf)[0]

    shape_full = fit_f.shape.numpy()
    shape_ext = gamma_extend(fit_t.shape.numpy()[None, :], dt, t_full[-1])[0][:t_full.size]
    tw_tr = twin_voxel(fit_t, lab_t, v, raw_tr, t_sim=t_full, shape_sim=shape_ext)
    tw_full = twin_voxel(fit_f, lab_f, v, raw_full)
    k_full, fit_full = ssvd_residue(truth, aif_full, dt)

    shape_tr = fit_t.shape.numpy()
    print(json.dumps({"patient": n, "voxel": [int(i) for i in v], "frames": int(raw_full.size), "keep": keep,
                      "fold": folds.get(n), "vendor_Tmax": float(s.maps["Tmax"][v]) if "Tmax" in s.maps else None,
                      "aif_shape_maxdiff_first_L": float(np.abs(shape_tr - shape_full[:shape_tr.size]).max()),
                      "twin_trunc_x": tw_tr["x"].tolist(), "twin_full_x": tw_full["x"].tolist(),
                      "twin_trunc_nmse": tw_tr["nmse"], "twin_full_nmse": tw_full["nmse"],
                      "tmax_ssvd_full": float(k_full.argmax() * dt), "tmax_ssvd_trunc": float(k_s.argmax() * dt),
                      "tmax_osvd_trunc": float(k_o.argmax() * dt)}, indent=1), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, patient=n, voxel=np.array(v), dt=dt, seconds=a.seconds, keep=keep,
        t_full=t_full, t70=t70, shape_ext=shape_ext, shape_tr=fit_t.shape.numpy(), truth=truth, truth_smooth=tw_full["smooth"], obs=obs,
        aif_full=aif_full, aif_tr=aif_tr, shape_full=shape_full,
        osvd_k=k_o, osvd_fit=fit_o, ssvd_k=k_s, ssvd_fit=fit_s, gamma=gam, learned=lrn,
        twin_tr_smooth=tw_tr["smooth"], twin_tr_fit=tw_tr["fit"], twin_tr_sim=tw_tr["sim"], twin_tr_q_sim=tw_tr["q_sim"],
        twin_full_fit=tw_full["fit"], twin_full_q=tw_full["q"],
        ssvd_full_k=k_full, ssvd_full_fit=fit_full)
    print("->", a.out)


if __name__ == "__main__":
    main()
