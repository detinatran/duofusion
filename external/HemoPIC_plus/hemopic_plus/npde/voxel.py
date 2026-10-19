"""Voxel-level transport model.

The ROI fit fixes what a single noisy voxel curve cannot identify: the Windkessel
relaxation (R, tau) and the flow F that drive the outflow q(t), the input
dispersion, the Peclet number and, for the neural PDE, the closure's split
(nu, phi) into a fast and a slow tube. A voxel curve identifies three things:
its amplitude (the area of C is Cin times that of (F/q) c_in, so Cin carries the
indicator-dilution CBV), its shape (the rate k = q / (60 V), i.e. the MTT) and
its arrival time. Every voxel therefore gets its own (log V, log Cin, delay),
with quadratic priors centred on its ROI (shrinkage towards the regional fit):

    L_v = nMSE_v + lam_v (log V - log V_roi)^2 + lam_c (log Cin - log Cin_roi)^2
                 + lam_d (delay - delay_roi)^2,

where nMSE_v is HemoPIC's weighted normalised squared error of the voxel curve
(its denominator is floored at a fraction of the ROI curve energy, so empty
voxels fall back on the prior). It is minimised voxel by voxel with a batched
Levenberg-Marquardt (three parameters, finite-difference Jacobian). Maps:
MTT_v = 60 V_v / CBF_roi, CBV_v = CBV_roi Cin_v / Cin_roi and CBF_v = 60 CBV_v / MTT_v,
where CBF_roi is the tail mean of the ROI outflow (as for the ROI level of the maps). The generator of a tube depends on Pe only, so
its eigen-modes are computed once per ROI and every voxel curve is the exact
modal recursion of transport.advance, batched over voxels. A single tube with
one cell (n_cells = 1) is HemoPIC+'s well-mixed compartment.
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from ..uq.torch_model import DTYPE, interp_shift
from . import transport as TP


def smooth_curves(y, sigma_frames, baseline_frames, clip_nonneg):
    """signal.smooth_curve_gaussian applied to every row of y (N, T)."""
    y = np.asarray(y, dtype=np.float64)
    bf = int(min(max(int(baseline_frames), 0), y.shape[1]))
    base = y[:, :bf].mean(axis=1, keepdims=True) if bf >= 1 else np.zeros((y.shape[0], 1))
    sig = float(sigma_frames) if np.isfinite(sigma_frames) and sigma_frames > 0 else 1.5
    ys = gaussian_filter1d(y - base, sigma=sig, axis=1, mode="nearest") + base
    return np.maximum(ys, 0.0) if clip_nonneg else ys


def roi_modes(log_pe, n_cells):
    """Eigenvalues lam and weights wgt (each (n,)) of one tube, per unit rate k."""
    lp = torch.as_tensor([float(log_pe)], dtype=DTYPE)
    lam, wgt = TP.modes(*TP.tridiag(lp, int(n_cells)))
    return lam[0].detach(), wgt[0].detach()


def tissue_modes(kdt, u, lam, wgt, nu, phi):
    """Tissue curve of parallel tubes that share the modes (lam, wgt).
    kdt, u: (B, T); nu, phi: (M,) volume and flow fractions. Same recursion as
    transport.advance with tube m running at rate k phi_m / nu_m."""
    B, T = kdt.shape
    kk = kdt[:, None, :] * (phi / nu)[None, :, None]
    g = wgt / lam
    y = torch.zeros(B, nu.shape[0], lam.shape[0], dtype=DTYPE)
    out = [torch.zeros(B, dtype=DTYPE)]
    for k in range(T - 1):
        e = torch.exp(kk[:, :, k, None] * lam)
        y = e * y + g * (e - 1.0) * u[:, k, None, None]
        out.append((nu[None, :] * y.sum(dim=2)).sum(dim=1))
    return torch.stack(out, dim=1)


class RoiModel:
    """Everything shared by the voxels of one ROI. theta_roi: the eight transport
    parameters (log R, log tau, log V, log F, log Cin, delay, log disp, log Pe)."""

    def __init__(self, cfg, t, dt, shape, theta_roi, n_cells=24, nu=None, phi=None):
        self.cfg, self.dt = cfg, float(dt)
        self.t = torch.as_tensor(t, dtype=DTYPE)
        self.shape = torch.as_tensor(shape, dtype=DTYPE)
        self.theta = torch.as_tensor(theta_roi, dtype=DTYPE)
        self.lam, self.wgt = roi_modes(self.theta[7], n_cells)
        self.nu = torch.ones(1, dtype=DTYPE) if nu is None else torch.as_tensor(nu, dtype=DTYPE)
        self.phi = torch.ones(1, dtype=DTYPE) if phi is None else torch.as_tensor(phi, dtype=DTYPE)
        self.disperse = float(np.exp(-self.dt / float(torch.exp(self.theta[6])))) > 1e-12

    FREE = (2, 4, 5)

    def full_theta(self, x):
        """x (B, 3) = (log V, log Cin, delay) -> (B, 8)."""
        th = self.theta[None, :].expand(x.shape[0], -1)
        return torch.cat([th[:, :2], x[:, 0:1], th[:, 3:4], x[:, 1:3], th[:, 6:]], dim=1)

    @property
    def centre(self):
        return self.theta[list(self.FREE)]

    def simulate(self, x):
        th = self.full_theta(x)
        shape_in = interp_shift(self.shape, self.t, th[:, 5])
        if self.disperse:
            shape_in = TP.disperse_rows(shape_in, self.dt, torch.exp(th[:, 6]))
        q_out, kdt, u = TP.drive(self.cfg, self.t[None, :], self.dt, shape_in, th)
        return q_out, tissue_modes(kdt, u, self.lam, self.wgt, self.nu, self.phi)

    def roi_curve(self):
        with torch.no_grad():
            return self.simulate(self.centre[None, :])[1][0]


def fit_voxels(model, c, w, lam_v=0.02, lam_c=0.02, lam_d=0.002, den_floor=0.05, span=float(np.log(10.0)),
               iters=30, chunk=16384, grid_step=1.0, tol=1e-7):
    """Fit (log V, log Cin, delay) of every voxel curve c (N, T) of one ROI by a
    batched, box-constrained Levenberg-Marquardt on the residuals of L_v
    (finite-difference Jacobian, no autograd). The delay starts from the best
    point of a 1 s grid over the delay bounds, V and Cin from the ROI.
    Returns x (N, 3), the loss and the data term (nMSE) per voxel."""
    c = torch.as_tensor(c, dtype=DTYPE)
    w = torch.as_tensor(w, dtype=DTYPE)
    N, T = c.shape
    lo_d, hi_d = (float(b) for b in model.cfg.delay_bounds)
    centre = model.centre
    lo = torch.stack([centre[0] - span, centre[1] - span, torch.as_tensor(lo_d, dtype=DTYPE)])
    hi = torch.stack([centre[0] + span, centre[1] + span, torch.as_tensor(hi_d, dtype=DTYPE)])
    sq_prior = torch.sqrt(torch.as_tensor([lam_v, lam_c, lam_d], dtype=DTYPE))
    roi = model.roi_curve()
    floor = den_floor * float((w * roi ** 2).mean())
    h = torch.as_tensor([1e-4, 1e-4, 1e-3], dtype=DTYPE)
    grid = torch.arange(lo_d, hi_d + 1e-9, float(grid_step), dtype=DTYPE)
    x_out = torch.empty(N, 3, dtype=DTYPE)
    loss_out = torch.empty(N, dtype=DTYPE)
    data_out = torch.empty(N, dtype=DTYPE)

    def resid(x, cc, sw):
        """(n, T + 3) residuals whose squared sum is L_v; returns them and the data part."""
        _, cf = model.simulate(x)
        rd = sw * (cf - cc)
        return torch.cat([rd, sq_prior * (x - centre)], dim=1), (rd ** 2).sum(dim=1)

    with torch.no_grad():
        for s in range(0, N, chunk):
            cc = c[s:s + chunk]
            n = cc.shape[0]
            sw = torch.sqrt(w[None, :] / (T * torch.clamp((w[None, :] * cc ** 2).mean(dim=1, keepdim=True), min=floor)))
            x = centre[None, :].repeat(n, 1)
            best = torch.full((n,), float("inf"), dtype=DTYPE)
            for d in grid:
                xd = x.clone()
                xd[:, 2] = d
                L = (resid(xd, cc, sw)[0] ** 2).sum(dim=1)
                better = L < best
                best = torch.where(better, L, best)
                x[better, 2] = d
            r, D = resid(x, cc, sw)
            L = (r ** 2).sum(dim=1)
            mu = torch.full((n,), 1e-3, dtype=DTYPE)
            active = torch.ones(n, dtype=torch.bool)
            for it in range(iters):
                ia = torch.nonzero(active).squeeze(1)
                if ia.numel() == 0:
                    break
                xa, ra, ca, swa = x[ia], r[ia], cc[ia], sw[ia]
                J = torch.stack([(resid(xa + h[k] * torch.eye(3, dtype=DTYPE)[k], ca, swa)[0] - ra) / h[k]
                                 for k in range(3)], dim=2)
                JTJ = J.transpose(1, 2) @ J
                g = (J.transpose(1, 2) @ ra[:, :, None])[:, :, 0]
                A = JTJ + mu[ia, None, None] * torch.diag_embed(torch.diagonal(JTJ, dim1=1, dim2=2) + 1e-9)
                step = torch.linalg.solve(A, -g[:, :, None])[:, :, 0]
                xn = torch.minimum(torch.maximum(xa + step, lo), hi)
                rn, Dn = resid(xn, ca, swa)
                Ln = (rn ** 2).sum(dim=1)
                acc = Ln < L[ia]
                gain = torch.where(acc, (L[ia] - Ln) / torch.clamp(L[ia], min=1e-30), torch.zeros_like(Ln))
                ib = ia[acc]
                x[ib], r[ib], L[ib], D[ib] = xn[acc], rn[acc], Ln[acc], Dn[acc]
                mu[ia] = torch.where(acc, mu[ia] * 0.3, mu[ia] * 10.0).clamp(1e-7, 1e7)
                active[ia] = ~((acc & (gain < tol)) | (mu[ia] >= 1e7))
            x_out[s:s + n], loss_out[s:s + n], data_out[s:s + n] = x, L, D
    return x_out.numpy(), loss_out.numpy(), data_out.numpy()


def roi_cbf(model, tail_frac):
    """CBF level of the ROI: tail mean of its Windkessel outflow (as for the ROI level of the maps)."""
    with torch.no_grad():
        q_out, _ = model.simulate(model.centre[None, :])
    q = q_out[0].numpy()
    tail = int(max(3, int(np.floor(q.size * float(tail_frac)))))
    return float(max(q[q.size - tail:].mean(), 1e-9))


def voxel_maps(model, x, tail_frac):
    """CBF, CBV and MTT of fitted voxels x (N, 3) = (log V, log Cin, delay), anchored
    on the ROI: CBF_roi is the tail mean of the ROI outflow and CBV_roi its V."""
    cbf_roi = roi_cbf(model, tail_frac)
    v_roi, cin_roi = float(torch.exp(model.theta[2])), float(model.theta[4])
    mtt = 60.0 * np.exp(x[:, 0]) / cbf_roi
    cbv = v_roi * np.exp(x[:, 1] - cin_roi)
    return 60.0 * cbv / mtt, cbv, mtt


def _scaled_weights(model, c, w, den_floor):
    T = c.shape[1]
    floor = den_floor * float((w * model.roi_curve() ** 2).mean())
    den = torch.clamp((w[None, :] * c ** 2).mean(dim=1, keepdim=True), min=floor)
    return torch.sqrt(w[None, :] / (T * den)), den[:, 0]


def _residuals(model, x, c, sw, sq_prior, centre):
    _, cf = model.simulate(x)
    return torch.cat([sw * (cf - c), sq_prior * (x - centre)], dim=1), cf


def _jacobian(model, x, c, sw, sq_prior, centre, r):
    h = torch.as_tensor([1e-4, 1e-4, 1e-3], dtype=DTYPE)
    eye = torch.eye(3, dtype=DTYPE)
    return torch.stack([(_residuals(model, x + h[k] * eye[k], c, sw, sq_prior, centre)[0] - r) / h[k] for k in range(3)], dim=2)


def voxel_laplace(model, x, c, w, lam_v=0.02, lam_c=0.02, lam_d=0.002, den_floor=0.05, sigma_s=1.8, dof=0, chunk=16384):
    """Tempered Laplace posterior of the voxel parameters, read like the ROI posterior:
    p(x | c) is proportional to exp(-kappa_v L_v) with kappa_v = n_eff D_v / (2 N_v), where
    N_v is the weighted squared residual of the voxel at its fit, D_v its (floored)
    signal energy and n_eff the effectively independent frames among those with
    weight (dof > 0: n_eff - dof, the residual degrees of freedom of the voxel fit).
    The covariance is the inverse of the Gauss-Newton Hessian 2 kappa_v J^T J
    of the stacked data and prior residuals (box bounds ignored).
    Returns the covariance (N, 3, 3), kappa_v (N,) and the prior standard deviations
    (N, 3) under the same temperature."""
    c = torch.as_tensor(c, dtype=DTYPE)
    x = torch.as_tensor(x, dtype=DTYPE)
    w = torch.as_tensor(w, dtype=DTYPE)
    N = c.shape[0]
    lam = torch.as_tensor([lam_v, lam_c, lam_d], dtype=DTYPE)
    sq_prior = torch.sqrt(lam)
    centre = model.centre
    n_eff = float((w > 0).sum()) / (2.0 * np.sqrt(np.pi) * float(sigma_s)) - float(dof)
    cov = torch.empty(N, 3, 3, dtype=DTYPE)
    kap = torch.empty(N, dtype=DTYPE)
    with torch.no_grad():
        for s in range(0, N, chunk):
            cc, xx = c[s:s + chunk], x[s:s + chunk]
            sw, den = _scaled_weights(model, cc, w, den_floor)
            r, cf = _residuals(model, xx, cc, sw, sq_prior, centre)
            J = _jacobian(model, xx, cc, sw, sq_prior, centre, r)
            num = (w[None, :] * (cf - cc) ** 2).mean(dim=1)
            k_v = n_eff * den / (2.0 * torch.clamp(num, min=1e-30))
            H = 2.0 * k_v[:, None, None] * (J.transpose(1, 2) @ J)
            cov[s:s + cc.shape[0]] = torch.linalg.inv(H + 1e-12 * torch.eye(3, dtype=DTYPE))
            kap[s:s + cc.shape[0]] = k_v
    prior_sd = 1.0 / torch.sqrt(2.0 * kap[:, None] * lam[None, :])
    return cov.numpy(), kap.numpy(), prior_sd.numpy()


def roi_sensitivity(make_model, theta, x, c, w, tail_frac, lam_v=0.02, lam_c=0.02, lam_d=0.002, den_floor=0.05,
                    step=1e-4, chunk=16384):
    """Linear response of the voxel fit to the ROI parameters (implicit function
    theorem with Gauss-Newton blocks): dx/dtheta = -(J_x^T J_x)^-1 J_x^T J_theta, where
    J_theta is the change of the voxel's stacked residuals (data and priors, whose
    centres are ROI parameters) when one ROI parameter moves. make_model(theta)
    builds the RoiModel of the ROI. Returns dx/dtheta (N, 3, P) and
    d log CBF_roi / dtheta (P,)."""
    theta = np.asarray(theta, dtype=np.float64)
    P = theta.size
    c = torch.as_tensor(c, dtype=DTYPE)
    x = torch.as_tensor(x, dtype=DTYPE)
    w = torch.as_tensor(w, dtype=DTYPE)
    N = c.shape[0]
    sq_prior = torch.sqrt(torch.as_tensor([lam_v, lam_c, lam_d], dtype=DTYPE))
    base = make_model(theta)
    moved = []
    for j in range(P):
        th = theta.copy()
        th[j] += step
        moved.append(make_model(th))
    lcbf0 = np.log(roi_cbf(base, tail_frac))
    dlcbf = np.array([(np.log(roi_cbf(m, tail_frac)) - lcbf0) / step for m in moved])
    out = torch.empty(N, 3, P, dtype=DTYPE)
    with torch.no_grad():
        for s in range(0, N, chunk):
            cc, xx = c[s:s + chunk], x[s:s + chunk]
            sw, _ = _scaled_weights(base, cc, w, den_floor)
            r0, _ = _residuals(base, xx, cc, sw, sq_prior, base.centre)
            Jx = _jacobian(base, xx, cc, sw, sq_prior, base.centre, r0)
            Jt = torch.stack([(_residuals(m, xx, cc, sw, sq_prior, m.centre)[0] - r0) / step for m in moved], dim=2)
            JtJ = Jx.transpose(1, 2) @ Jx + 1e-12 * torch.eye(3, dtype=DTYPE)
            out[s:s + cc.shape[0]] = -torch.linalg.solve(JtJ, Jx.transpose(1, 2) @ Jt)
    return out.numpy(), dlcbf
