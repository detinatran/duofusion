"""Neural closure of the transport PDE: transit-time heterogeneity.

Indicator-dilution theory requires tracer kinetics to be linear in the
concentration and to conserve mass, and the maps use the central volume theorem
(MTT = 60 V / CBF). One tube (npde/transport.py) can only make the transit
times narrower than HemoPIC's tank; ROIs that mix fast and slow tissue (for
example a lesion inside a healthy ROI) need a wider spread. The closure lets
each ROI split into two parallel tubes with blood-volume fractions (nu, 1 - nu)
and flow fractions (phi, 1 - phi). Whatever the split, mass is conserved and the
mean transit time stays 1/k, so CBF, CBV and MTT keep their physical meaning;
only the shape of the transit-time distribution changes.

A single curve cannot identify (nu, phi) next to seven physical parameters, so
a small network shared by all ROIs and patients predicts them from ROI
descriptors that use neither the lesion label nor any reference map (tissue
class, tracer priors, time to peak and width of the curve relative to the
arterial input). It is trained jointly with the per-ROI physics of the training
patients and frozen on held-out patients. It starts close to the plain PDE
(nu = 0.5, phi = 0.57: rates 1.13 and 0.87 of the mean), and a small penalty on
logit(nu) - logit(phi) pulls the two rates together unless the data need them apart.

Cohort batching: ROIs of different patients have different frame counts and
TRs; curves are padded to the longest one with zero weight.
"""
from dataclasses import dataclass

import numpy as np
import torch

from ..uq.data import load_subject
from ..uq.torch_model import DTYPE
from . import transport as TP

TISSUES = ("GM", "WM", "CSF", "BrainStem")


@dataclass
class Cohort:
    subject: np.ndarray
    roi_ids: np.ndarray
    tissue: list
    t: torch.Tensor
    dt: torch.Tensor
    shape: torch.Tensor
    c_obs: torch.Tensor
    w: torch.Tensor
    n_frames: np.ndarray
    log_f0: torch.Tensor
    log_v0: torch.Tensor
    log_cin0: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor
    theta0: torch.Tensor
    z: torch.Tensor
    cfg: object

    def take(self, idx):
        idx = np.asarray(idx)
        ti = torch.as_tensor(idx, dtype=torch.long)
        return Cohort(self.subject[idx], self.roi_ids[idx], [self.tissue[i] for i in idx], self.t[ti], self.dt[ti],
                      self.shape[ti], self.c_obs[ti], self.w[ti], self.n_frames[idx], self.log_f0[ti], self.log_v0[ti],
                      self.log_cin0[ti], self.lo[ti], self.hi[ti], self.theta0[ti], self.z[ti], self.cfg)


def _curve_features(c, shape, t):
    """Time to peak relative to the arterial curve (s) and width ratio (FWHM)."""
    def fwhm(x):
        m = x.max()
        above = np.where(x >= 0.5 * m)[0]
        return (above[-1] - above[0] + 1) if above.size else 1
    dt = t[1] - t[0]
    return (np.argmax(c) - np.argmax(shape)) * dt, fwhm(c) / max(fwhm(shape), 1)


def load_cohort(fit_dirs, pde_root=None, pe_bounds=(0.02, 100.0)):
    """Stack the ROIs of several fitted subjects. theta0 comes from
    <pde_root>/<subject>/pde_theta.npz when present (plain PDE fit)."""
    rows, Tmax, cfg = [], 0, None
    for fd in fit_dirs:
        b = load_subject(fd)
        cfg = b.cfg
        th0 = torch.cat([b.theta_fit, torch.full((len(b.roi_ids), 1), float(np.log(3.0)), dtype=DTYPE)], 1)
        if pde_root is not None and (pde_root / fd.name / "pde_theta.npz").exists():
            zz = np.load(pde_root / fd.name / "pde_theta.npz")
            pos = {int(r): i for i, r in enumerate(zz["roi_ids"])}
            for k, rid in enumerate(b.roi_ids):
                if int(rid) in pos:
                    th0[k] = torch.as_tensor(zz["theta"][pos[int(rid)]], dtype=DTYPE)
        lo, hi = TP.bounds(b, pe_bounds)
        rows.append((int(fd.name.split("_")[1]), b, th0, lo, hi))
        Tmax = max(Tmax, b.t.shape[0])
    B = sum(len(r[1].roi_ids) for r in rows)
    t = torch.zeros(B, Tmax, dtype=DTYPE)
    shape = torch.zeros(B, Tmax, dtype=DTYPE)
    c_obs = torch.zeros(B, Tmax, dtype=DTYPE)
    w = torch.zeros(B, Tmax, dtype=DTYPE)
    dt = torch.zeros(B, dtype=DTYPE)
    subj, rids, tis, nfr, f0, v0, c0, LO, HI, TH, feats = [], [], [], [], [], [], [], [], [], [], []
    i = 0
    for pid, b, th0, lo, hi in rows:
        T = b.t.shape[0]
        tt = b.t.numpy()
        sh = b.shape.numpy()
        for k in range(len(b.roi_ids)):
            t[i] = torch.as_tensor(tt[0] + b.dt * np.arange(Tmax), dtype=DTYPE)
            shape[i, :T] = b.shape
            shape[i, T:] = b.shape[-1]
            c_obs[i, :T] = b.c_obs[k]
            w[i, :T] = b.w[k]
            dt[i] = b.dt
            ttp, wr = _curve_features(b.c_obs[k].numpy(), sh, tt)
            feats.append([float(b.tissue[k] == s) for s in TISSUES]
                         + [float(b.log_f0[k]), float(b.log_v0[k]), float(b.log_v0[k] - b.log_f0[k]), ttp, np.log(wr)])
            subj.append(pid)
            rids.append(int(b.roi_ids[k]))
            tis.append(b.tissue[k])
            nfr.append(T)
            i += 1
        f0.append(b.log_f0)
        v0.append(b.log_v0)
        c0.append(b.log_cin0)
        LO.append(lo)
        HI.append(hi)
        TH.append(th0)
    feats = np.array(feats)
    mu, sd = feats.mean(0), feats.std(0) + 1e-9
    mu[:len(TISSUES)], sd[:len(TISSUES)] = 0.0, 1.0
    z = torch.as_tensor((feats - mu) / sd, dtype=DTYPE)
    th0 = torch.cat(TH)
    lo, hi = torch.cat(LO), torch.cat(HI)
    th0 = torch.minimum(torch.maximum(th0, lo + 1e-6), hi - 1e-6)
    return Cohort(np.array(subj), np.array(rids), tis, t, dt, shape, c_obs, w, np.array(nfr), torch.cat(f0), torch.cat(v0),
                  torch.cat(c0), lo, hi, th0, z, cfg)


def shift_rows(shape, t, dt, delay):
    """Per-row np.interp(t - delay, t, shape) with edge extension."""
    B, T = shape.shape
    x = ((t - delay[:, None]) - t[:, :1]) / dt[:, None]
    x = x.clamp(0.0, T - 1.0)
    i0 = x.floor().clamp(max=T - 2).long()
    frac = x - i0.to(x.dtype)
    return torch.gather(shape, 1, i0) * (1.0 - frac) + torch.gather(shape, 1, i0 + 1) * frac


class Closure(torch.nn.Module):
    """MLP z -> (nu, phi) of two parallel tubes, both within [0.05, 0.95]."""

    def __init__(self, n_z, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_z, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, 2))
        torch.nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0.0, 0.3]))
        self.double()

    def asymmetry(self, z):
        """logit(nu) - logit(phi): zero when both tubes run at the same rate (plain PDE)."""
        h = self.net(z)
        return h[:, 0] - h[:, 1]

    def forward(self, z):
        h = self.net(z)
        nu1 = 0.05 + 0.9 * torch.sigmoid(h[:, 0])
        phi1 = 0.05 + 0.9 * torch.sigmoid(h[:, 1])
        return torch.stack([nu1, 1.0 - nu1], 1), torch.stack([phi1, 1.0 - phi1], 1)


def simulate(co, theta, n_cells=24, paths=None):
    shape_in = TP.disperse_rows(shift_rows(co.shape, co.t, co.dt, theta[:, 5]), co.dt, torch.exp(theta[:, 6]))
    q_out, kdt, u = TP.drive(co.cfg, co.t, co.dt[:, None], shape_in, theta)
    return q_out, TP.tissue(kdt, u, theta[:, 7], n_cells, paths)


def objective(co, theta, n_cells=24, paths=None):
    _, c_fit = simulate(co, theta, n_cells, paths)
    num = (co.w * (c_fit - co.c_obs) ** 2).sum(1)
    den = (co.w * co.c_obs ** 2).sum(1) + 1e-12
    data = num / den
    return data + TP.priors(co.cfg, theta, co.log_f0, co.log_v0, co.log_cin0), data, c_fit


def to_theta(co, x):
    return co.lo + (co.hi - co.lo) * torch.sigmoid(x)


def from_theta(co, theta):
    p = ((theta - co.lo) / (co.hi - co.lo)).clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


def fit(co, closure=None, train_closure=False, steps=1500, lr=0.02, lr_net=2e-3, shape_penalty=1e-3, n_cells=24,
        log_every=0, seed=0):
    """Adam on the per-ROI physics (and, if train_closure, on the shared network).
    Returns theta, per-ROI objective and data term, fitted curves and the loss history."""
    torch.manual_seed(seed)
    x = from_theta(co, co.theta0).clone().requires_grad_(True)
    groups = [{"params": [x], "lr": lr}]
    if closure is not None and train_closure:
        groups.append({"params": list(closure.parameters()), "lr": lr_net})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []
    for it in range(steps):
        opt.zero_grad()
        theta = to_theta(co, x)
        paths = closure(co.z) if closure is not None else None
        obj, data, _ = objective(co, theta, n_cells, paths)
        loss = obj.mean()
        if closure is not None and train_closure:
            loss = loss + shape_penalty * (closure.asymmetry(co.z) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
        if log_every and it % log_every == 0:
            print(f"    step {it:5d} loss {hist[-1]:.5f} median data {float(data.detach().median()):.5f}", flush=True)
    with torch.no_grad():
        theta = to_theta(co, x)
        paths = closure(co.z) if closure is not None else None
        obj, data, c_fit = objective(co, theta, n_cells, paths)
    return theta.detach(), obj, data, c_fit, hist
