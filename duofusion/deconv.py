"""Baseline deconvolutions for the truncation study: standard SVD (sSVD) and gamma-variate
extrapolation of truncated curves before deconvolution."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def ssvd_maps(curves, aif, dt, k_h=0.73, rho=1.04, threshold=0.2, chunk=20000):
    """Standard truncated SVD (lower-triangular Toeplitz AIF matrix, singular values below
    threshold * max set to zero). curves (N, T), aif (T,). Returns CBF (ml/100 g/min) and
    Tmax (s, time of the residue maximum)."""
    T = aif.size
    A = np.zeros((T, T))
    for i in range(T):
        A[i, : i + 1] = aif[i::-1] * dt
    U, S, Vt = np.linalg.svd(A)
    inv = np.where(S >= threshold * S.max(), 1.0 / S, 0.0)
    P = (Vt.T * inv) @ U.T                                     # pseudo-inverse
    scale = 100.0 * k_h / rho
    cbf = np.empty(curves.shape[0])
    tmax = np.empty(curves.shape[0])
    for s0 in range(0, curves.shape[0], chunk):
        k = curves[s0:s0 + chunk] @ P.T
        cbf[s0:s0 + chunk] = scale * 60.0 * k.max(axis=1)
        tmax[s0:s0 + chunk] = k.argmax(axis=1) * dt
    return cbf, tmax


def gamma_extend(curves, dt, total_s, smooth=1.0, low=0.1, tail=0.5):
    """Extend (N, T) first-pass curves to total_s seconds with a gamma variate
    K (t - t0)^a exp(-(t - t0) / b), fitted per curve by weighted linear least squares on
    log C over the frames from bolus arrival (C > low * peak) to the first frame after the
    peak where C < tail * peak (or the last frame when the scan ends before that).
    Measured frames are kept; only the missing ones are filled. Curves whose fit fails
    (a <= 0 or b <= 0) are continued by an exponential decay from their last frames."""
    N, T = curves.shape
    Te = int(np.floor(total_s / dt)) + 1
    if Te <= T:
        return curves.copy()
    c = gaussian_filter1d(curves, smooth, axis=1, mode="nearest") if smooth > 0 else curves
    t = np.arange(T) * dt
    p = c.argmax(axis=1)
    m = c[np.arange(N), p]
    above = c > low * m[:, None]
    # arrival: last frame before the peak below low * peak
    before = np.where((np.arange(T)[None, :] < p[:, None]) & ~above, np.arange(T)[None, :], -1).max(axis=1)
    t0 = np.maximum(before, 0) * dt
    after_low = (np.arange(T)[None, :] > p[:, None]) & (c < tail * m[:, None])
    end = np.where(after_low.any(axis=1), after_low.argmax(axis=1), T - 1)
    idx = np.arange(T)[None, :]
    W = (idx > before[:, None]) & (idx <= end[:, None]) & above & (c > 0)
    tau = np.maximum(t[None, :] - t0[:, None], 1e-3)
    X = np.stack([np.ones((N, T)), np.log(tau), -tau], axis=2)          # (N, T, 3)
    y = np.log(np.maximum(c, 1e-9))
    Wf = W.astype(float)
    XtX = np.einsum("nt,nti,ntj->nij", Wf, X, X) + 1e-6 * np.eye(3)
    Xty = np.einsum("nt,nti,nt->ni", Wf, X, y)
    ok = W.sum(axis=1) >= 4
    beta = np.zeros((N, 3))
    beta[ok] = np.linalg.solve(XtX[ok], Xty[ok][..., None])[..., 0]
    a, inv_b = beta[:, 1], beta[:, 2]
    good = ok & (a > 0) & (inv_b > 0) & (m > 0)
    te = np.arange(Te) * dt
    taue = np.maximum(te[None, :] - t0[:, None], 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        g = np.exp(beta[:, 0:1] + a[:, None] * np.log(np.maximum(taue, 1e-9)) - inv_b[:, None] * taue)
    g = np.where(taue > 0, g, 0.0)
    out = np.zeros((N, Te))
    out[:, :T] = curves
    # fallback: exponential decay from the mean of the last three frames, time constant 8 s
    last = curves[:, -3:].mean(axis=1)
    decay = last[:, None] * np.exp(-(te[None, T:] - t[-1]) / 8.0)
    out[:, T:] = np.where(good[:, None] & np.isfinite(g[:, T:]), g[:, T:], decay)
    return np.maximum(out, 0.0)
