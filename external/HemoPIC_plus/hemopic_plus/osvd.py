"""Independent reference: block-circulant SVD deconvolution with a per-voxel
oscillation-index threshold (oSVD, Wu et al., MRM 2003), driven by the same
automatic AIF as [C6].

Used only for evaluation (scripts/osvd_check.py): if oSVD with our AIF also
disagrees with the ISLES deconvolution maps across patients, the CVT gap of
HemoPIC+ comes from the AIF / processing of the reference maps rather than
from the HemoPIC+ model.

Units: c = (kH/rho) * CBF * (AIF (*) R). With k(t) = CBF R(t) from the
deconvolution, CBF = 100 kH/rho * 60 * max k [ml/100g/min],
CBV = 100 kH/rho * AUC(c)/AUC(AIF) [ml/100g], MTT = 60 CBV / CBF [s].
"""
import numpy as np


def _circulant_svd(aif, dt):
    n = aif.size
    a = np.concatenate([aif, np.zeros(n)])
    m = 2 * n
    idx = (np.arange(m)[:, None] - np.arange(m)[None, :]) % m
    D = dt * a[idx]
    U, S, Vt = np.linalg.svd(D)
    return U, S, Vt


def _oscillation(k):
    """Wu et al. oscillation index of residue functions k (n_vox, L)."""
    L = k.shape[1]
    kmax = np.maximum(np.max(k, axis=1), 1e-12)
    second = np.abs(k[:, 2:] - 2 * k[:, 1:-1] + k[:, :-2]).sum(axis=1)
    return second / (L * kmax)


def osvd_maps(curves, aif, dt, k_h=0.73, rho=1.04, osc_target=0.095,
              thresholds=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5), chunk=20000, return_tmax=False):
    """curves: (n_vox, T) baseline-subtracted concentration; aif: (T,).
    Returns cbf, cbv, mtt arrays of shape (n_vox,), and Tmax (time of the
    residue maximum, seconds) when return_tmax is set."""
    n_vox, T = curves.shape
    U, S, Vt = _circulant_svd(aif, dt)
    smax = float(S.max())
    scale = 100.0 * k_h / rho
    auc_a = max(float(np.sum(aif) * dt), 1e-12)
    cbf = np.full(n_vox, np.nan)
    tmax = np.full(n_vox, np.nan)
    for s0 in range(0, n_vox, chunk):
        c = curves[s0:s0 + chunk]
        cpad = np.concatenate([c, np.zeros_like(c)], axis=1).T
        UTc = U.T @ cpad
        chosen = np.full(c.shape[0], np.nan)
        arg = np.full(c.shape[0], np.nan)
        for thr in thresholds:
            inv = np.where(S >= thr * smax, 1.0 / S, 0.0)
            k = (Vt.T @ (inv[:, None] * UTc)).T
            ok = np.isnan(chosen) & (_oscillation(k) <= osc_target)
            chosen[ok] = np.max(k[ok], axis=1)
            arg[ok] = np.argmax(k[ok][:, :T], axis=1)
        if np.any(np.isnan(chosen)):
            inv = np.where(S >= thresholds[-1] * smax, 1.0 / S, 0.0)
            k = (Vt.T @ (inv[:, None] * UTc)).T
            rest = np.isnan(chosen)
            chosen[rest] = np.max(k[rest], axis=1)
            arg[rest] = np.argmax(k[rest][:, :T], axis=1)
        cbf[s0:s0 + chunk] = scale * 60.0 * chosen
        tmax[s0:s0 + chunk] = arg * dt
    cbv = scale * np.sum(curves, axis=1) * dt / auc_a
    with np.errstate(divide="ignore", invalid="ignore"):
        mtt = np.where(cbf > 0, 60.0 * cbv / cbf, np.nan)
    if return_tmax:
        return cbf, cbv, mtt, tmax
    return cbf, cbv, mtt
