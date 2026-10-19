"""Posterior of the ROI parameters around the HemoPIC(+) fit.

Posterior. fit_roi_curve minimises L(theta) = D(theta) + R(theta), with D the
weighted normalised MSE and R the quadratic log-space priors. We read L as a
negative log posterior with one temperature per ROI,

    p(theta | c) ∝ exp(-kappa * L(theta)),   kappa = n_eff * den / (2 * num_hat),

which is exactly a Gaussian likelihood with noise variance num_hat / w_t (num_hat
= weighted mean squared residual at the fit) and n_eff independent samples;
the posterior mode is the HemoPIC(+) fit itself and the priors keep the
relative strength they have in the original objective. The ROI curves are
smoothed with a Gaussian of sigma_s frames, so consecutive frames are not
independent: n_eff = T / (2 sqrt(pi) sigma_s).

Laplace: Gaussian approximation N(theta_hat, (kappa H)^-1) on the parameters
that are not at a bound; parameters at a bound are reported as pinned.
"""
import numpy as np
import torch
from scipy.optimize import minimize

from .torch_model import DTYPE, derived, objective

PIN_TOL = 1e-3


def n_eff(batch):
    T = batch.c_obs.shape[1]
    return T / (2.0 * np.sqrt(np.pi) * float(batch.cfg.smooth_sigma_frames))


def refine_map(batch, maxiter=200):
    """Polish the Powell solution with L-BFGS-B on exact gradients (bounds kept).
    A ROI keeps its Powell solution unless the objective strictly improves."""
    theta = torch.minimum(torch.maximum(batch.theta_fit, batch.lo), batch.hi).clone()
    for i in range(theta.shape[0]):
        def f(x):
            th = torch.as_tensor(x, dtype=DTYPE).reshape(1, -1).requires_grad_(True)
            val = objective(batch, th, roi_index=[i])[0][0]
            val.backward()
            return float(val.detach()), th.grad[0].detach().numpy().astype(np.float64)
        x0 = theta[i].numpy()
        f0, _ = f(x0)
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(batch.lo[i].numpy(), batch.hi[i].numpy())),
                       options={"maxiter": maxiter})
        if np.isfinite(res.fun) and res.fun < f0 - 1e-12:
            theta[i] = torch.as_tensor(res.x, dtype=DTYPE)
    return theta


def temperature(batch, theta):
    with torch.no_grad():
        _, _, num, den = objective(batch, theta)
    return n_eff(batch) * den / (2.0 * torch.clamp(num, min=1e-30))


def prior_sd(batch, kappa):
    """Prior standard deviation of every log-parameter under exp(-kappa L)
    (uniform prior inside the bounds where the objective has no penalty)."""
    cfg = batch.cfg
    B, P = batch.lo.shape
    lam = [cfg.reg_lambda_logr, 0.0, cfg.lambda_v, cfg.lambda_f, cfg.lambda_in]
    if P >= 7:
        lam += [cfg.lambda_delay, cfg.lambda_dispersion]
    if P == 8:
        lam += [0.0]
    out = torch.empty(B, P, dtype=DTYPE)
    for k, lam_k in enumerate(lam):
        if lam_k > 0:
            out[:, k] = 1.0 / torch.sqrt(2.0 * kappa * float(lam_k))
        else:
            out[:, k] = (batch.hi[:, k] - batch.lo[:, k]) / np.sqrt(12.0)
        out[:, k] = torch.minimum(out[:, k], (batch.hi[:, k] - batch.lo[:, k]) / np.sqrt(12.0))
    return out


def laplace(batch, theta, kappa, eig_floor=1e-8):
    """Returns cov (B,P,P) in log space (zero rows/cols for pinned params),
    pinned mask (B,P) and the eigenvalues of the free Hessian block."""
    B, P = theta.shape
    width = batch.hi - batch.lo
    pinned = ((theta - batch.lo) < PIN_TOL * width) | ((batch.hi - theta) < PIN_TOL * width)
    cov = torch.zeros(B, P, P, dtype=DTYPE)
    eigs = []
    for i in range(B):
        def f(th):
            return objective(batch, th.reshape(1, -1), roi_index=[i])[0][0]
        H = torch.autograd.functional.hessian(f, theta[i].clone()) * kappa[i]
        free = torch.where(~pinned[i])[0]
        Hf = H[free][:, free]
        Hf = 0.5 * (Hf + Hf.T)
        ev, V = torch.linalg.eigh(Hf)
        eigs.append(ev.numpy())
        ev_c = torch.clamp(ev, min=eig_floor * max(float(ev.abs().max()), 1e-30))
        Cf = (V / ev_c) @ V.T
        cov[i][free.unsqueeze(1), free.unsqueeze(0)] = Cf
    return cov, pinned, eigs


def sample_gaussian(batch, theta, cov, n=2000, seed=0):
    """Draws from N(theta, cov) truncated to the bounds (rejection, then clip)."""
    g = torch.Generator().manual_seed(seed)
    B, P = theta.shape
    out = torch.empty(B, n, P, dtype=DTYPE)
    for i in range(B):
        C = cov[i] + 1e-14 * torch.eye(P, dtype=DTYPE)
        ev, V = torch.linalg.eigh(C)
        L = V * torch.sqrt(torch.clamp(ev, min=0.0))
        z = torch.randn(4 * n, P, generator=g, dtype=DTYPE)
        s = theta[i] + z @ L.T
        ok = ((s >= batch.lo[i]) & (s <= batch.hi[i])).all(dim=1)
        s = s[ok][:n]
        if s.shape[0] < n:
            extra = theta[i] + torch.randn(n - s.shape[0], P, generator=g, dtype=DTYPE) @ L.T
            s = torch.cat([s, torch.minimum(torch.maximum(extra, batch.lo[i]), batch.hi[i])])
        out[i] = s
    return out


def derived_samples(batch, samples):
    """Derived quantities for samples (B, S, P) -> dict of (B, S) tensors."""
    B, S, P = samples.shape
    out = {}
    with torch.no_grad():
        for i in range(B):
            d = derived(batch, samples[i])
            for k, v in d.items():
                out.setdefault(k, torch.empty(B, S, dtype=DTYPE))[i] = v
    return out
