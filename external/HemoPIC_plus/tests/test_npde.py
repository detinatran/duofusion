"""The transport PDE must contain HemoPIC+ (tank + [C2] input kernel) as special
cases and keep the physics that defines the maps:
  1. N = 1 cell (any Pe) gives the HemoPIC+ curve exactly;
  2. Pe -> 0 (any N) gives the HemoPIC+ curve;
  3. with a constant inlet, every variant (parallel paths, unequal cells and
     dispersion) fills to the same steady state (mass conservation);
  4. the modal solver equals the matrix-exponential reference;
  5. transit-time densities integrate to 1 with mean 1 (in units of MTT),
     for one tube, learned coefficient fields and parallel paths.

python tests/test_npde.py  (uses outputs/<config>/training_22 if present)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hemopic_plus.npde import transport as TP  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE, simulate  # noqa: E402


def with_pe(th7, log_pe):
    return torch.cat([th7, torch.full((th7.shape[0], 1), float(log_pe), dtype=DTYPE)], 1)


def check(fit_dir):
    b = load_subject(fit_dir)
    B = len(b.roi_ids)
    g = torch.Generator().manual_seed(0)
    th7 = b.theta_fit.clone()
    with torch.no_grad():
        _, c_ref = simulate(b, th7)
        scale = c_ref.abs().max()
        e1 = float((TP.simulate(b, with_pe(th7, 3.0), n_cells=1)[1] - c_ref).abs().max() / scale)
        e2 = float((TP.simulate(b, with_pe(th7, np.log(1e-5)), n_cells=24)[1] - c_ref).abs().max() / scale)

        b2 = b.subset(range(B))
        b2.shape = torch.ones_like(b.shape)
        th = with_pe(th7, np.log(5.0))
        th[:, 1] = b.lo[:, 1]
        target = None
        e3 = 0.0
        nu = torch.rand(B, 3, generator=g, dtype=DTYPE) + 1.0
        phi = torch.rand(B, 3, generator=g, dtype=DTYPE) + 1.0
        vol = torch.rand(B, 16, generator=g, dtype=DTYPE) + 0.2
        for kw in ({}, {"paths": (nu / nu.sum(1, keepdim=True), phi / phi.sum(1, keepdim=True))},
                   {"vol": vol / vol.sum(1, keepdim=True), "dmul": torch.rand(B, 15, generator=g, dtype=DTYPE) * 3}):
            q, c = TP.simulate(b2, th, n_cells=16, **kw)
            if target is None:
                target = torch.exp(th[:, 3]) / q[:, -1] * torch.exp(th[:, 4])
            e3 = max(e3, float(((c[:, -1] - target).abs() / target).max()))

        from hemopic_plus.uq.torch_model import interp_shift
        vol24 = torch.rand(B, 24, generator=g, dtype=DTYPE) + 0.2
        vol24 = vol24 / vol24.sum(1, keepdim=True)
        dm24 = torch.rand(B, 23, generator=g, dtype=DTYPE) * 3 + 0.1
        e4 = 0.0
        for lpe in (-3.0, 0.0, 2.0, np.log(100.0)):
            th = with_pe(th7, lpe)
            _, kdt, u = TP.drive(b.cfg, b.t[None, :], b.dt, interp_shift(b.shape, b.t, th[:, 5]), th)
            for v, d in ((None, None), (vol24, dm24)):
                G, vv = TP.operator(th[:, 7], 24, v, d)
                ref = TP.advance_expm(kdt, u, G, vv)
                new = TP.advance(kdt, u, *TP.tridiag(th[:, 7], 24, v, d))
                e4 = max(e4, float((new - ref).abs().max() / ref.abs().max()))

        s = np.linspace(0.0, 30.0, 3001)
        lp = torch.tensor([1.0], dtype=DTYPE)
        hs = [TP.transit_density(lp, 24, s)[0], TP.transit_density(lp, 24, s, vol24[:1], dm24[:1])[0],
              TP.transit_density_paths(lp, 24, s, torch.tensor([[0.3, 0.7]], dtype=DTYPE),
                                       torch.tensor([[0.8, 0.2]], dtype=DTYPE))[0]]
        e5 = 0.0
        for h in hs:
            h = h.numpy()
            area = np.trapezoid(h, s)
            e5 = max(e5, abs(area - 1), abs(np.trapezoid(h * s, s) / area - 1))
    assert e1 < 1e-10, e1
    assert e2 < 1e-4, e2
    assert e3 < 5e-3, e3
    assert e4 < 1e-6, e4
    assert e5 < 2e-3, e5
    return e1, e2, e3, e4, e5


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "outputs"
    for d in [root / "full" / "training_22", root / "full_aif" / "training_22"]:
        if d.exists():
            print(d, "N=1 %.1e, Pe->0 %.1e, steady state %.1e, modal vs expm %.1e, transit densities %.1e" % check(d))
    print("all tests passed")
