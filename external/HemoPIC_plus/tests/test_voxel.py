"""Voxel-level transport model (npde/voxel.py):
  1. a voxel at the ROI parameters reproduces the ROI curve of the transport
     model (tank with N = 1, PDE with parallel tubes), using the ROI's modes;
  2. the batched Levenberg-Marquardt recovers (log V, log Cin, delay) of
     synthetic voxel curves with 2% noise, and the maps (CBF, CBV, MTT) follow
     (ROIs whose MTT is above three sampling intervals; a shorter MTT is not
     resolved by the data);
  3. a voxel equal to its ROI gets the ROI's CBF, CBV and MTT;
  4. voxel uncertainty: the tempered Laplace intervals (dof 3) cover the truth of
     synthetic voxels with smoothed noise, and the implicit-function response
     dx/dtheta matches refits after a small change of an ROI parameter.

python tests/test_voxel.py  (uses outputs/full_aif/training_22 if present)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from hemopic_plus.npde import transport as TP  # noqa: E402
from hemopic_plus.npde.voxel import RoiModel, fit_voxels, roi_sensitivity, voxel_laplace, voxel_maps  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE  # noqa: E402


def check(fit_dir):
    b = load_subject(fit_dir)
    t, dt, shape, w = b.t.numpy(), b.dt, b.shape.numpy(), b.w[0].numpy()
    rng = np.random.default_rng(0)
    g = torch.Generator().manual_seed(0)
    e1 = e2 = e3 = 0.0
    mtt = 60.0 * torch.exp(b.theta_fit[:, 2] - b.theta_fit[:, 3])
    rois = [i for i in range(len(b.roi_ids)) if float(mtt[i]) > 3 * dt][:4]
    for i in rois:
        th7 = b.theta_fit[i]
        nu = torch.rand(2, generator=g, dtype=DTYPE) + 0.5
        phi = torch.rand(2, generator=g, dtype=DTYPE) + 0.5
        nu, phi = nu / nu.sum(), phi / phi.sum()
        for th, n, paths in ((torch.cat([th7, torch.zeros(1, dtype=DTYPE)]), 1, None),
                             (torch.cat([th7, torch.tensor([2.0], dtype=DTYPE)]), 24, (nu, phi))):
            m = RoiModel(b.cfg, t, dt, shape, th.numpy(), n_cells=n,
                         nu=None if paths is None else nu, phi=None if paths is None else phi)
            with torch.no_grad():
                ref = TP.simulate(b.subset([i]), th[None, :], n_cells=n,
                                  paths=None if paths is None else (nu[None, :], phi[None, :]))[1][0]
            e1 = max(e1, float((m.roi_curve() - ref).abs().max() / ref.abs().max()))
            cen = m.centre.numpy()
            xt = cen + np.c_[rng.normal(0, .3, 200), rng.normal(0, .3, 200), rng.uniform(-2, 5, 200)]
            xt[:, 2] = np.clip(xt[:, 2], *b.cfg.delay_bounds)
            with torch.no_grad():
                ct = m.simulate(torch.as_tensor(xt))[1].numpy()
            cn = ct + rng.normal(0, 0.02 * ct.max(1, keepdims=True), ct.shape)
            x, _, _ = fit_voxels(m, cn, w, lam_v=1e-6, lam_c=1e-6, lam_d=1e-6)
            mt, me = voxel_maps(m, xt, b.cfg.tail_frac), voxel_maps(m, x, b.cfg.tail_frac)
            e2 = max(e2, max(float(np.median(np.abs(a / b_ - 1))) for a, b_ in zip(me, mt)))
            f0, v0, m0 = voxel_maps(m, cen[None, :], b.cfg.tail_frac)
            with torch.no_grad():
                q = m.simulate(m.centre[None, :])[0][0].numpy()
            tail = int(max(3, int(np.floor(q.size * float(b.cfg.tail_frac)))))
            q_mean = q[q.size - tail:].mean()
            v_roi = float(torch.exp(th[2]))
            e3 = max(e3, abs(f0[0] / q_mean - 1), abs(v0[0] / v_roi - 1), abs(m0[0] / (60 * v_roi / q_mean) - 1))
    assert e1 < 1e-8, e1
    assert e2 < 0.05, e2
    assert e3 < 1e-12, e3
    return e1, e2, e3


def check_uq(fit_dir):
    b = load_subject(fit_dir)
    t, dt, shape, w = b.t.numpy(), b.dt, b.shape.numpy(), b.w[0].numpy()
    cfg = b.cfg
    rng = np.random.default_rng(1)
    mtt = 60.0 * torch.exp(b.theta_fit[:, 2] - b.theta_fit[:, 3])
    i = next(i for i in range(len(b.roi_ids)) if float(mtt[i]) > 3 * dt)
    th = np.r_[b.theta_fit[i].numpy(), 0.0]

    def make(th_):
        return RoiModel(cfg, t, dt, shape, th_, n_cells=1)
    m = make(th)
    cen = m.centre.numpy()
    n = 300
    xt = cen + np.c_[rng.normal(0, .3, n), rng.normal(0, .3, n), rng.uniform(-2, 5, n)]
    xt[:, 2] = np.clip(xt[:, 2], *cfg.delay_bounds)
    with torch.no_grad():
        ct = m.simulate(torch.as_tensor(xt))[1].numpy()
    e = gaussian_filter1d(rng.standard_normal(ct.shape), cfg.smooth_sigma_frames, axis=1, mode="nearest")
    cn = ct + 0.03 * ct.max(1, keepdims=True) * e / e.std(1, keepdims=True)
    lam = dict(lam_v=1e-6, lam_c=1e-6, lam_d=1e-6)
    x, _, _ = fit_voxels(m, cn, w, **lam)
    cov, _, _ = voxel_laplace(m, x, cn, w, sigma_s=cfg.smooth_sigma_frames, dof=3, **lam)
    sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
    cover = float(np.mean(np.abs(x[:, :2] - xt[:, :2]) <= 1.959964 * sd[:, :2]))
    x0, _, _ = fit_voxels(m, cn, w)
    dx, _ = roi_sensitivity(make, th, x0, cn, w, cfg.tail_frac)
    err = 0.0
    for j in (0, 3):
        h = 0.01
        th_j = th.copy()
        th_j[j] += h
        xj, _, _ = fit_voxels(make(th_j), cn, w)
        emp = (xj[:, 0] - x0[:, 0]) / h
        err = max(err, float(np.median(np.abs(emp - dx[:, 0, j])) / np.median(np.abs(dx[:, 0, j]))))
    assert 0.85 <= cover <= 0.995, cover
    assert err < 0.1, err
    return cover, err


if __name__ == "__main__":
    d = Path(__file__).resolve().parent.parent / "outputs" / "full_aif" / "training_22"
    if d.exists():
        print(d, "ROI curve %.1e, synthetic recovery (median rel. error of the maps) %.3f, ROI anchoring %.1e" % check(d))
        print("voxel Laplace 95%% coverage (dof 3) %.3f, dx/dtheta against refits (median rel. error) %.3f" % check_uq(d))
    print("all tests passed")
