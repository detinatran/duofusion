"""ROI batch for the uncertainty tools (uq/laplace.py, uq/mcmc.py, the checks)
with the transport-PDE tissue model and a fixed neural closure.

uq.torch_model.simulate / objective / derived dispatch to this object when
is_pde is set, so Laplace, MCMC and the held-out and semi-synthetic checks run
unchanged on the eight PDE parameters (log R, log tau, log V, log F, log Cin,
delay, log dispersion, log Pe). The two-tube split (nu, phi) of every ROI is
held fixed at the value the frozen network predicts.
"""
import numpy as np
import torch

from ..uq.torch_model import DTYPE
from ..uq.torch_model import derived as tank_derived
from . import transport as TP


class PDEBatch:
    is_pde = True

    def __init__(self, b, theta, nu=None, phi=None, n_cells=24, pe_bounds=(0.02, 100.0)):
        self.b = b
        for k in ("roi_ids", "tissue", "c_obs", "w", "shape", "t", "dt", "log_f0", "log_v0", "log_cin0", "cfg"):
            setattr(self, k, getattr(b, k))
        for k in ("info", "w_curve_raw", "c_fit_numpy"):
            if hasattr(b, k):
                setattr(self, k, getattr(b, k))
        self.lo, self.hi = TP.bounds(b, pe_bounds)
        self.pe_bounds = pe_bounds
        theta = torch.as_tensor(theta, dtype=DTYPE)
        self.theta_fit = torch.minimum(torch.maximum(theta, self.lo), self.hi)
        self.nu = None if nu is None else torch.as_tensor(nu, dtype=DTYPE)
        self.phi = None if phi is None else torch.as_tensor(phi, dtype=DTYPE)
        self.n_cells = int(n_cells)

    @property
    def P(self):
        return 8

    @property
    def names(self):
        return TP.NAMES

    def subset(self, idx):
        idx = list(idx)
        ti = torch.as_tensor(idx, dtype=torch.long)
        out = PDEBatch(self.b.subset(idx), self.theta_fit[ti], None if self.nu is None else self.nu[ti],
                       None if self.phi is None else self.phi[ti], self.n_cells, self.pe_bounds)
        for k in ("info", "w_curve_raw"):
            if hasattr(self, k):
                setattr(out, k, getattr(self, k))
        return out

    def _paths(self, n_rows, roi_index):
        if self.nu is None:
            return None
        idx = torch.arange(n_rows) if roi_index is None else torch.as_tensor(roi_index, dtype=torch.long)
        return self.nu[idx], self.phi[idx]

    def pde_simulate(self, theta, roi_index=None):
        return TP.simulate(self, theta, self.n_cells, self._paths(theta.shape[0], roi_index))

    def pde_objective(self, theta, roi_index=None, w=None):
        idx = torch.arange(theta.shape[0]) if roi_index is None else torch.as_tensor(roi_index, dtype=torch.long)
        _, c_fit = self.pde_simulate(theta, roi_index)
        c_obs = self.c_obs[idx]
        ww = self.w[idx] if w is None else w
        num = (ww * (c_fit - c_obs) ** 2).mean(dim=1)
        den = (ww * c_obs ** 2).mean(dim=1) + 1e-12
        data = num / den
        return data + TP.priors(self.cfg, theta, self.log_f0[idx], self.log_v0[idx], self.log_cin0[idx]), data, num, den

    def pde_derived(self, theta):
        return tank_derived(self.b, theta[:, :7])


def load_npde_subject(fit_dir, npde_dir, member=None):
    """PDEBatch of one subject: curves and priors from the HemoPIC+ fit folder,
    PDE parameters and the closure's split from <name>__npde/<subject>/npde_theta.npz.
    member selects one bootstrap network of the ensemble (None = the main one)."""
    from ..uq.data import load_subject
    b = load_subject(fit_dir)
    z = np.load(npde_dir / "npde_theta.npz")
    assert list(z["roi_ids"]) == list(b.roi_ids)
    if member is None:
        theta, nu, phi = z["theta"], z["nu"], z["phi"]
    else:
        theta, nu, phi = z["theta_members"][member], z["nu_members"][member], z["phi_members"][member]
    n_cells = int(z["n_cells"]) if "n_cells" in z else 24
    return PDEBatch(b, theta, nu, phi, n_cells)
