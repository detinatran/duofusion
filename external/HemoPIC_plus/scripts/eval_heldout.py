"""Check 1: hold out part of each curve and test the predictive intervals.

python scripts/eval_heldout.py configs/full.yaml [--folds 5]

For every ROI the time frames are split into `folds` contiguous blocks. Each
block in turn is hidden (weight 0), the posterior is refitted on the remaining
frames (MAP + Laplace, same temperature rule as run_uq.py), and the predictive
distribution of the hidden frames is formed from parameter draws plus the
residual noise. A calibrated model covers ~50/80/95% of the hidden values with
its central 50/80/95% intervals. Also reported: the same coverage with the
noise term only (no parameter uncertainty), to show how much the parameter
uncertainty contributes.

Writes <out_root>/<name>/eval_uq/heldout.csv and prints the cohort coverage.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import list_patients, subject_id  # noqa: E402
from hemopic_plus.uq import laplace as LP  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE, objective, simulate  # noqa: E402

LEVELS = (0.5, 0.8, 0.95)


def fit_masked(b, i, theta0, w):
    def f(x):
        th = torch.as_tensor(x, dtype=DTYPE).reshape(1, -1).requires_grad_(True)
        val = objective(b, th, roi_index=[i], w=w[None, :])[0][0]
        val.backward()
        return float(val.detach()), th.grad[0].detach().numpy().astype(np.float64)
    res = minimize(f, theta0.numpy(), jac=True, method="L-BFGS-B",
                   bounds=list(zip(b.lo[i].numpy(), b.hi[i].numpy())), options={"maxiter": 300})
    return torch.as_tensor(res.x, dtype=DTYPE)


def laplace_one(b, i, theta, w, kappa):
    width = b.hi[i] - b.lo[i]
    pinned = ((theta - b.lo[i]) < LP.PIN_TOL * width) | ((b.hi[i] - theta) < LP.PIN_TOL * width)
    H = torch.autograd.functional.hessian(lambda th: objective(b, th.reshape(1, -1), roi_index=[i], w=w[None, :])[0][0],
                                          theta.clone()) * kappa
    free = torch.where(~pinned)[0]
    P = theta.shape[0]
    cov = torch.zeros(P, P, dtype=DTYPE)
    if free.numel():
        Hf = 0.5 * (H[free][:, free] + H[free][:, free].T)
        ev, V = torch.linalg.eigh(Hf)
        ev = torch.clamp(ev, min=1e-8 * max(float(ev.abs().max()), 1e-30))
        cov[free.unsqueeze(1), free.unsqueeze(0)] = (V / ev) @ V.T
    return cov


def run_subject(args):
    fit_dir, folds, n_draws = args
    torch.set_num_threads(1)
    return heldout_rows(load_subject(fit_dir), fit_dir.name, folds, n_draws)


def heldout_rows(b, name, folds=5, n_draws=400):
    theta_all = LP.refine_map(b)
    g = torch.Generator().manual_seed(0)
    B, T = b.c_obs.shape
    edges = np.linspace(0, T, folds + 1).astype(int)
    rows = []
    for i in range(B):
        for k in range(folds):
            hide = torch.zeros(T, dtype=torch.bool)
            hide[edges[k]:edges[k + 1]] = True
            w = b.w[i].clone()
            w[hide] = 0.0
            theta = fit_masked(b, i, theta_all[i], w)
            with torch.no_grad():
                _, c_fit = simulate(b, theta[None, :], roi_index=[i])
                r = c_fit[0] - b.c_obs[i]
                n_train = int((~hide).sum())
                sigma2 = float((w * r ** 2).sum() / n_train)
                den = float((w * b.c_obs[i] ** 2).mean()) + 1e-12
                num = float((w * r ** 2).mean())
                kappa = LP.n_eff(b) * (n_train / T) * den / (2.0 * max(num, 1e-30))
            cov = laplace_one(b, i, theta, w, kappa)
            draws = LP.sample_gaussian(b.subset([i]), theta[None, :], cov[None], n=n_draws, seed=k)[0]
            with torch.no_grad():
                _, c_draw = simulate(b, draws, roi_index=[i] * draws.shape[0])
            wt = b.w[i][hide]
            noise_sd = torch.sqrt(sigma2 / torch.clamp(wt, min=1e-6))
            pred = c_draw[:, hide] + noise_sd[None, :] * torch.randn(n_draws, int(hide.sum()), generator=g, dtype=DTYPE)
            obs = b.c_obs[i][hide]
            for lev in LEVELS:
                lo_q, hi_q = (1 - lev) / 2, 1 - (1 - lev) / 2
                lo = torch.quantile(pred, lo_q, dim=0)
                hi = torch.quantile(pred, hi_q, dim=0)
                z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(hi_q, dtype=DTYPE)))
                lo_n = c_fit[0][hide] - z * noise_sd
                hi_n = c_fit[0][hide] + z * noise_sd
                rows.append({"subject": name, "roi_id": int(b.roi_ids[i]), "tissue": b.tissue[i], "fold": k,
                             "level": lev, "n": int(hide.sum()),
                             "covered": float(((obs >= lo) & (obs <= hi)).double().mean()),
                             "covered_noise_only": float(((obs >= lo_n) & (obs <= hi_n)).double().mean())})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--draws", type=int, default=400)
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("out_root", a.out_root),) if v})
    root = cfg.resolve("out_root") / cfg.name
    pids = a.patients or list_patients(root)
    jobs = [(root / subject_id(pid), a.folds, a.draws) for pid in pids if (root / subject_id(pid) / "fit_curves.npz").exists()]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers or os.cpu_count() or 1) as ex:
        for r in ex.map(run_subject, jobs):
            rows.extend(r)
    out = root / "eval_uq"
    out.mkdir(exist_ok=True)
    with open(out / "heldout.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[{cfg.name}] held-out coverage over {len(jobs)} subjects (weighted by hidden frames)")
    for lev in LEVELS:
        sel = [r for r in rows if r["level"] == lev]
        n = np.array([r["n"] for r in sel])
        cov = np.sum(n * np.array([r["covered"] for r in sel])) / n.sum()
        cov_n = np.sum(n * np.array([r["covered_noise_only"] for r in sel])) / n.sum()
        print(f"  nominal {lev:.0%}: posterior predictive {cov:.1%}   noise only {cov_n:.1%}")


if __name__ == "__main__":
    main()
