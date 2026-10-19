"""Differentiable, batched copy of the ROI model in hemopic_plus/fit.py.

Every ROI of one subject (and every MCMC chain) is evaluated in one tensor
operation; the time recursion stays a Python loop over <= 80 frames. The loss
is the exact objective minimised by fit_roi_curve (weighted normalised MSE +
quadratic log-space priors), so the MAP of the posterior defined in
uq/laplace.py is the HemoPIC(+) fit itself. tests/test_uq_torch.py checks the
curves against the numpy implementation.
"""
from dataclasses import dataclass

import numpy as np
import torch

DTYPE = torch.float64
NAMES5 = ["log_R", "log_tau", "log_V", "log_F", "log_Cin"]
NAMES7 = NAMES5 + ["delay", "log_disp"]


@dataclass
class RoiBatch:
    """Everything needed to evaluate the objective of B ROIs of one subject."""
    roi_ids: np.ndarray
    tissue: list
    c_obs: torch.Tensor
    w: torch.Tensor
    shape: torch.Tensor
    t: torch.Tensor
    dt: float
    log_f0: torch.Tensor
    log_v0: torch.Tensor
    log_cin0: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor
    theta_fit: torch.Tensor
    cfg: object

    @property
    def P(self):
        return self.lo.shape[1]

    @property
    def names(self):
        return NAMES7 if self.P == 7 else NAMES5

    def subset(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        sel = idx.numpy()
        return RoiBatch(self.roi_ids[sel], [self.tissue[i] for i in sel], self.c_obs[idx], self.w[idx], self.shape,
                        self.t, self.dt, self.log_f0[idx], self.log_v0[idx], self.log_cin0[idx], self.lo[idx],
                        self.hi[idx], self.theta_fit[idx], self.cfg)


def weight_from_curve(a):
    """(HemoPIC) weight_from_aif: 0.2 + 0.8 * min-max normalised curve."""
    w = a - a.min()
    mx = w.max()
    if mx > 0:
        w = w / mx
    return 0.2 + 0.8 * np.clip(w, 0.0, 1.0)


def bounds_for(cfg, t, dt, f0, v0, cin0):
    """Log-space bounds, identical to fit_roi_curve."""
    t_end = float(t[-1]) if t.size >= 2 else float(dt)
    tau_min = max(float(dt), 0.001)
    tau_max = min(float(cfg.tau_cap_seconds), 0.98 * t_end)
    if not tau_max > tau_min:
        tau_max = tau_min + 20.0 * float(dt)
    f_lo = max(f0 * 0.20, 1e-6)
    f_hi = max(f0 * 5.0, f_lo * 1.10)
    v_lo = max(v0 * 0.20, 1e-6)
    v_hi = max(v0 * 5.0, v_lo * 1.10)
    c_lo = max(cin0 * 0.10, 1e-6)
    c_hi = max(cin0 * 10.0, c_lo * 1.10)
    lo = [np.log(1e-10), np.log(tau_min), np.log(v_lo), np.log(f_lo), np.log(c_lo)]
    hi = [np.log(float(cfg.r_max)), np.log(tau_max), np.log(v_hi), np.log(f_hi), np.log(c_hi)]
    if cfg.delay_dispersion:
        lo += [float(cfg.delay_bounds[0]), np.log(cfg.dispersion_bounds[0])]
        hi += [float(cfg.delay_bounds[1]), np.log(cfg.dispersion_bounds[1])]
    return np.array(lo), np.array(hi)


def interp_shift(shape, t, delay):
    """np.interp(t - delay, t, shape) with edge extension, batched over delay (N,)."""
    T = shape.shape[0]
    dt = t[1] - t[0]
    x = ((t[None, :] - delay[:, None]) - t[0]) / dt
    x = x.clamp(0.0, T - 1.0)
    i0 = x.floor().clamp(max=T - 2).long()
    frac = x - i0.to(x.dtype)
    s0 = shape[i0]
    s1 = shape[i0 + 1]
    return s0 * (1.0 - frac) + s1 * frac


def disperse(x, dt, tau_d):
    """Exact first-order recursive exponential filter (lfilter with zi=0) on x - x[0]."""
    a = torch.exp(-dt / tau_d)
    base = x[:, :1]
    u = x - base
    ys = [torch.zeros_like(u[:, 0])]
    for n in range(1, u.shape[1]):
        ys.append(a * ys[-1] + (1.0 - a) * u[:, n])
    return torch.stack(ys, dim=1) + base


def simulate(batch, theta, roi_index=None):
    """theta: (N, P) with N = len(roi_index) (defaults to one row per ROI).
    Returns q_out (N, T) and c_fit (N, T). Batches that carry their own tissue
    model (npde.uqbatch.PDEBatch) are dispatched to it."""
    if getattr(batch, "is_pde", False):
        return batch.pde_simulate(theta, roi_index)
    cfg = batch.cfg
    t = batch.t
    r, tau, v, f, cin = (torch.exp(theta[:, k]) for k in range(5))
    p0 = f if cfg.wk_init == "steady" else torch.full_like(f, float(cfg.p0))
    q_out = f[:, None] + (p0 / r - f)[:, None] * torch.exp(-t[None, :] / tau[:, None])
    N, T = q_out.shape
    if cfg.delay_dispersion:
        shape = interp_shift(batch.shape, t, theta[:, 5])
        shape = disperse(shape, batch.dt, torch.exp(theta[:, 6]))
    else:
        shape = batch.shape[None, :].expand(N, T)
    c_in = cin[:, None] * shape
    dt_eff = batch.dt / 60.0 if cfg.flow_per_minute else batch.dt
    q_floor = torch.clamp(float(cfg.q_floor_ratio) * f, min=1e-12)
    ct = [torch.zeros_like(f)]
    for n in range(T - 1):
        q = torch.clamp(q_out[:, n], min=0.0)
        q_eff = torch.maximum(q, q_floor)
        a = torch.exp(-(q_eff / v) * dt_eff)
        drive = (f / q_eff) * c_in[:, n]
        ct.append(ct[-1] * a + drive * (1.0 - a))
    return q_out, torch.stack(ct, dim=1)


def objective(batch, theta, roi_index=None, w=None):
    """Per-row value of the fit_roi_curve objective and its data part."""
    if getattr(batch, "is_pde", False):
        return batch.pde_objective(theta, roi_index, w)
    idx = torch.arange(theta.shape[0]) if roi_index is None else torch.as_tensor(roi_index)
    cfg = batch.cfg
    _, c_fit = simulate(batch, theta)
    c_obs = batch.c_obs[idx]
    ww = batch.w[idx] if w is None else w
    num = (ww * (c_fit - c_obs) ** 2).mean(dim=1)
    den = (ww * c_obs ** 2).mean(dim=1) + 1e-12
    data = num / den
    reg = (float(cfg.reg_lambda_logr) * theta[:, 0] ** 2
           + float(cfg.lambda_f) * (theta[:, 3] - batch.log_f0[idx]) ** 2
           + float(cfg.lambda_v) * (theta[:, 2] - batch.log_v0[idx]) ** 2
           + float(cfg.lambda_in) * (theta[:, 4] - batch.log_cin0[idx]) ** 2)
    if cfg.delay_dispersion:
        log_disp0 = float(np.log(max(cfg.dispersion_bounds[0], 1e-3)))
        reg = reg + float(cfg.lambda_delay) * theta[:, 5] ** 2 + float(cfg.lambda_dispersion) * (theta[:, 6] - log_disp0) ** 2
    return data + reg, data, num, den


def derived(batch, theta, roi_index=None):
    """Quantities that enter the maps: regional CBF level (mean of the outflow over
    the last tail_frac of the window, as in pipeline.py), V, MTT = 60 V / CBF, tau."""
    if getattr(batch, "is_pde", False):
        return batch.pde_derived(theta)
    cfg = batch.cfg
    q_out, _ = simulate(batch, theta)
    T = q_out.shape[1]
    tail = max(3, int(np.floor(T * float(cfg.tail_frac))))
    cbf = torch.clamp(q_out[:, T - tail:].mean(dim=1), min=float(cfg.eps_small))
    v = torch.exp(theta[:, 2])
    return {"CBF": cbf, "V": v, "MTT": 60.0 * v / cbf, "tau": torch.exp(theta[:, 1]), "F": torch.exp(theta[:, 3])}
