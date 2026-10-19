"""Batched random-walk Metropolis for the ROI posteriors exp(-kappa L(theta)).

All ROIs of a subject and all chains move in one tensor step. Proposals are
preconditioned with the Laplace covariance (pinned directions get a proposal
scale of 5% of their bound width), the per-ROI step size is adapted to an
acceptance rate of ~0.25 during burn-in, and halfway through burn-in the
proposal covariance is replaced by the empirical covariance of the chains.
The bounds act as a uniform prior (proposals outside are rejected).
"""
import numpy as np
import torch

from .torch_model import DTYPE, objective


def _logpost(batch, theta, roi_index, kappa_rows):
    lo = batch.lo[roi_index]
    hi = batch.hi[roi_index]
    inside = ((theta >= lo) & (theta <= hi)).all(dim=1)
    val = objective(batch, torch.minimum(torch.maximum(theta, lo), hi), roi_index=roi_index)[0]
    lp = -kappa_rows * val
    return torch.where(inside & torch.isfinite(lp), lp, torch.full_like(lp, -np.inf))


def _chol(C, P):
    C = 0.5 * (C + C.transpose(-1, -2)) + 1e-12 * torch.eye(P, dtype=DTYPE)
    ev, V = torch.linalg.eigh(C)
    return V * torch.sqrt(torch.clamp(ev, min=1e-14)).unsqueeze(-2)


def run(batch, theta0, cov, pinned, kappa, chains=4, n_burn=2000, n_keep=4000, thin=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    B, P = theta0.shape
    roi_index = torch.arange(B).repeat_interleave(chains)
    kappa_rows = kappa[roi_index]
    width = (batch.hi - batch.lo)
    C = cov.clone()
    for i in range(B):
        for k in torch.where(pinned[i])[0]:
            C[i, k, k] = (0.05 * width[i, k]) ** 2
    L = _chol(C, P)
    scale = torch.full((B,), 2.38 / np.sqrt(P), dtype=DTYPE)
    jitter = torch.randn(B * chains, P, generator=g, dtype=DTYPE)
    x = theta0[roi_index] + 0.1 * torch.einsum("npq,nq->np", L[roi_index], jitter)
    x = torch.minimum(torch.maximum(x, batch.lo[roi_index]), batch.hi[roi_index])
    lp = _logpost(batch, x, roi_index, kappa_rows)
    keep = []
    acc_window = torch.zeros(B, dtype=DTYPE)
    acc_total = torch.zeros(B, dtype=DTYPE)
    burn_hist = []
    n_total = n_burn + n_keep
    with torch.no_grad():
        for step in range(n_total):
            z = torch.randn(B * chains, P, generator=g, dtype=DTYPE)
            prop = x + scale[roi_index, None] * torch.einsum("npq,nq->np", L[roi_index], z)
            lp_prop = _logpost(batch, prop, roi_index, kappa_rows)
            u = torch.rand(B * chains, generator=g, dtype=DTYPE)
            accept = torch.log(u) < (lp_prop - lp)
            x = torch.where(accept[:, None], prop, x)
            lp = torch.where(accept, lp_prop, lp)
            acc_rows = accept.to(DTYPE).reshape(B, chains).mean(dim=1)
            if step < n_burn:
                acc_window += acc_rows
                burn_hist.append(x.reshape(B, chains, P).clone())
                if (step + 1) % 100 == 0:
                    rate = acc_window / 100.0
                    scale = scale * torch.exp(rate - 0.25)
                    acc_window.zero_()
                if step + 1 == n_burn // 2:
                    hist = torch.stack(burn_hist[n_burn // 4:], dim=2).reshape(B, -1, P)
                    emp = torch.stack([torch.cov(h.T) for h in hist]).reshape(B, P, P)
                    L = _chol(emp + 1e-6 * torch.diag_embed(width ** 2 * 1e-4), P)
                    scale = torch.full((B,), 2.38 / np.sqrt(P), dtype=DTYPE)
                    burn_hist = []
            else:
                acc_total += acc_rows
                if (step - n_burn) % thin == 0:
                    keep.append(x.reshape(B, chains, P).clone())
    samples = torch.stack(keep, dim=2)
    return samples, acc_total / n_keep


def rhat_ess(samples):
    """Split-R-hat and a simple bulk ESS per ROI and parameter. samples: (B,C,S,P)."""
    B, C, S, P = samples.shape
    h = S // 2
    sp = torch.cat([samples[:, :, :h], samples[:, :, h:2 * h]], dim=1)
    chain_mean = sp.mean(dim=2)
    chain_var = sp.var(dim=2, unbiased=True)
    W = chain_var.mean(dim=1)
    Bv = h * chain_mean.var(dim=1, unbiased=True)
    var_hat = (h - 1) / h * W + Bv / h
    rhat = torch.sqrt(var_hat / torch.clamp(W, min=1e-30))
    x = samples - samples.mean(dim=2, keepdim=True)
    var = (x ** 2).mean(dim=2)
    ess = torch.empty(B, P, dtype=DTYPE)
    for lag_max in [min(S - 1, 200)]:
        rho_sum = torch.zeros(B, P, dtype=DTYPE)
        alive = torch.ones(B, P, dtype=torch.bool)
        for lag in range(1, lag_max):
            r = (x[:, :, lag:] * x[:, :, :-lag]).mean(dim=2) / torch.clamp(var, min=1e-30)
            r = r.mean(dim=1)
            alive = alive & (r > 0.05)
            rho_sum = rho_sum + torch.where(alive, r, torch.zeros_like(r))
        ess = C * S / (1.0 + 2.0 * rho_sum)
    return rhat, ess
