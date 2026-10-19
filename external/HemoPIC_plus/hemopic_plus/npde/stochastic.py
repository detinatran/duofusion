"""Stochastic transport PDE: white noise at the inlet of the tube.

    dc = k(t) (-dc/ds + D d2c/ds2) dt,
    inlet (Danckwerts)  c - D dc/ds = g(t) (c_in(t) + sigma_u xi(t)),
    observed  y_n = int_0^1 c ds (t_n) + e_n.

The noise enters through the boundary condition, so the stochastic model keeps
the deterministic one's structure: mass is conserved for every realisation and
the mean transit time stays 1/k. It stands for what the input model misses
(delay, dispersion, bolus shape). Within a frame the noise is frozen like the
input, eta_n ~ N(0, sigma_u^2). The state starts at zero and the system is
linear, so the tissue curve is

    C = Phi (u + eta),

with Phi (T, T) the response of the tissue curve at frame n to a unit inlet
value during frame k < n (transport.advance with u = delta_k, summed over the
modes and parallel tubes). The measured curve is smoothed with a Gaussian of
sigma_s frames before fitting (S, the smoothing matrix), and so is the model:
with white measurement noise e ~ N(0, sigma_v^2 I),

    y = S (Phi (u + eta) + e) ~ N(S Phi u, S (sigma_u^2 Phi Phi^T + sigma_v^2 I) S^T + delta I),

a Gaussian process whose likelihood is exact (delta, 1e-3 of the largest
observed value squared, only keeps the covariance well conditioned). Frames
that are not observed are predicted by Gaussian conditioning. sigma_u = 0 is
the deterministic model with the same noise model. sigma_v is expressed
relative to the largest observed value. The inlet noise is either additive
(sigma_u relative to the largest inlet value) or relative to the inlet value of
the frame, eta_n ~ N(0, (sigma_u u_n)^2), an error of the bolus itself; with
corr_time > 0 it is an Ornstein-Uhlenbeck process of that correlation time
(seconds) instead of white.
"""
import numpy as np
import torch

from ..uq.torch_model import DTYPE, interp_shift
from . import transport as TP

LOG_SIGMA_BOUNDS = (float(np.log(1e-4)), float(np.log(2.0)))


def smoothing_matrix(T, sigma):
    """S (T, T) such that S y = gaussian_filter1d(y, sigma, mode='nearest')."""
    from scipy.ndimage import gaussian_filter1d
    return torch.as_tensor(gaussian_filter1d(np.eye(T), float(sigma), axis=0, mode="nearest"), dtype=DTYPE)


def inlet(batch, theta):
    """Rate k dt (B, T) and inlet concentration u (B, T) of the deterministic model."""
    shape_in = TP.disperse_rows(interp_shift(batch.shape, batch.t, theta[:, 5]), batch.dt, torch.exp(theta[:, 6]))
    _, kdt, u = TP.drive(batch.cfg, batch.t[None, :], batch.dt, shape_in, theta)
    return kdt, u


def response(kdt, log_pe, n_cells, nu=None, phi=None):
    """Phi (B, T, T): Phi[n, k] is the tissue value at frame n per unit inlet value
    during frame k (zero for n <= k). nu, phi: (B, M) parallel tubes or None."""
    B, T = kdt.shape
    lam, wgt = TP.modes(*TP.tridiag(log_pe, int(n_cells)))
    if nu is None:
        nu = torch.ones(B, 1, dtype=DTYPE)
        phi = torch.ones(B, 1, dtype=DTYPE)
    kk = kdt[:, None, :] * (phi / nu)[:, :, None]
    g = (wgt / lam)[:, None, :, None]
    x = kk[:, :, None, :] * lam[:, None, :, None]
    L = torch.cat([torch.zeros_like(x[..., :1]), torch.cumsum(x, dim=-1)], dim=-1)
    a = g * torch.expm1(x)
    n = torch.arange(T)
    k = torch.arange(T)
    expo = L[..., :T, None] - L[..., None, 1:T + 1]
    mask = (n[:, None] > k[None, :]).to(DTYPE)
    expo = torch.where(mask.bool(), expo, torch.zeros_like(expo))
    term = a[..., None, :] * torch.exp(expo) * mask
    return (nu[:, :, None, None, None] * term).sum(dim=(1, 2))


class StochasticModel:
    """Gaussian-process view of the stochastic PDE for a batch of ROIs.
    batch: RoiBatch (tank, n_cells = 1, log Pe ignored) or PDEBatch."""

    def __init__(self, batch, n_cells=None, nu=None, phi=None, noise="relative", corr_time=0.0):
        self.b = batch
        self.noise = noise
        T0 = batch.c_obs.shape[1]
        lag = torch.arange(T0, dtype=DTYPE)
        rho = float(np.exp(-float(batch.dt) / corr_time)) if corr_time > 0 else 0.0
        self.C = rho ** (lag[:, None] - lag[None, :]).abs() if rho > 0 else torch.eye(T0, dtype=DTYPE)
        self.n_cells = int(getattr(batch, "n_cells", 1) if n_cells is None else n_cells)
        self.nu = getattr(batch, "nu", None) if nu is None else nu
        self.phi = getattr(batch, "phi", None) if phi is None else phi
        T = batch.c_obs.shape[1]
        self.S = smoothing_matrix(T, batch.cfg.smooth_sigma_frames)
        self.SS = self.S @ self.S.T
        self.y_scale = batch.c_obs.abs().max(dim=1).values.clamp(min=1e-12)

    def full_theta(self, theta):
        return theta if theta.shape[1] == 8 else torch.cat([theta, torch.full_like(theta[:, :1], -20.0)], 1)

    def parts(self, theta, idx):
        """Phi (B, T, T) and u (B, T) of the ROIs idx."""
        th = self.full_theta(theta)
        sub = _Rows(self.b, idx)
        kdt, u = inlet(sub, th)
        nu = None if self.nu is None else self.nu[idx]
        phi = None if self.phi is None else self.phi[idx]
        return response(kdt, th[:, 7], self.n_cells, nu, phi), u

    def mean(self, Phi, u):
        return (self.S[None] @ Phi @ u[:, :, None])[:, :, 0]

    def covariance(self, Phi, u, log_su, log_sv, idx):
        if self.noise == "relative":
            scale = torch.exp(log_su)[:, None] * u
        else:
            scale = (torch.exp(log_su) * u.detach().abs().max(dim=1).values.clamp(min=1e-12))[:, None].expand_as(u)
        sv = torch.exp(log_sv) * self.y_scale[idx]
        SP = self.S[None] @ Phi
        T = Phi.shape[1]
        nug = (1e-3 * self.y_scale[idx]) ** 2
        Q = scale[:, :, None] * self.C[None] * scale[:, None, :]
        return (SP @ Q @ SP.transpose(1, 2) + sv[:, None, None] ** 2 * self.SS[None]
                + nug[:, None, None] * torch.eye(T, dtype=DTYPE)[None])

    def loglik(self, theta, log_su, log_sv, idx, obs):
        """Log marginal likelihood of the observed frames obs (bool (T,) or (B, T))."""
        Phi, u = self.parts(theta, idx)
        mu = self.mean(Phi, u)
        K = self.covariance(Phi, u, log_su, log_sv, idx)
        y = self.b.c_obs[idx]
        out = []
        for r in range(len(idx)):
            o = obs if obs.dim() == 1 else obs[r]
            Ko = K[r][o][:, o]
            d = y[r, o] - mu[r, o]
            Lc = torch.linalg.cholesky(Ko)
            z = torch.cholesky_solve(d[:, None], Lc)[:, 0]
            out.append(-0.5 * (d @ z) - torch.log(torch.diagonal(Lc)).sum() - 0.5 * o.sum() * np.log(2 * np.pi))
        return torch.stack(out)

    def predict(self, theta, log_su, log_sv, idx, obs):
        """Mean and covariance of the noisy curve at every frame given the observed frames."""
        Phi, u = self.parts(theta, idx)
        mu = self.mean(Phi, u)
        K = self.covariance(Phi, u, log_su, log_sv, idx)
        y = self.b.c_obs[idx]
        means, covs = [], []
        for r in range(len(idx)):
            o = obs if obs.dim() == 1 else obs[r]
            Koo = K[r][o][:, o]
            Kao = K[r][:, o]
            A = torch.linalg.solve(Koo, Kao.T).T
            means.append(mu[r] + A @ (y[r, o] - mu[r, o]))
            covs.append(K[r] - A @ Kao.T)
        return torch.stack(means), torch.stack(covs), mu


class _Rows:
    """The rows idx of a batch, with only what inlet() needs."""

    def __init__(self, b, idx):
        ti = torch.as_tensor(idx, dtype=torch.long)
        self.shape, self.t, self.dt, self.cfg = b.shape, b.t, b.dt, b.cfg
        if self.shape.dim() == 2:
            self.shape = self.shape[ti]
