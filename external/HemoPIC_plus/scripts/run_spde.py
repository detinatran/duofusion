"""Stochastic transport PDE (npde/stochastic.py) against the same model without inlet noise.

python scripts/run_spde.py configs/full_aif.yaml [--models tank npde] [--folds 5] [--workers 24]

For every ROI two versions are fitted by maximum a posteriori on the exact
Gaussian likelihood, with the priors of the deterministic fit at its temperature
(kappa = n_eff D / (2 N) at that fit):
  det   sigma_u = 0: the deterministic model, smoothed like the data, with white
        measurement noise of fitted level sigma_v;
  spde  inlet noise sigma_u fitted as well. By default theta is that of det and
        only sigma_u and sigma_v are fitted (two stages: the physical fit, then
        its error model); --joint fits theta with them, which lets the noise
        replace the dynamics.
Models: tank (HemoPIC+'s compartment, <name>/) and npde (the neural PDE of the
held-out fold, <name>__npde/, nu and phi fixed). Reported per ROI: the gain in
log likelihood, sigma_u, sigma_v, the nRMSE of the noise-free curve, and the
predictive coverage of hidden blocks of time frames (each of --folds contiguous
blocks in turn is hidden, both versions are refitted on the rest and predict it
by Gaussian conditioning; 50/80/95% central intervals, median log score per frame).
--noise and --corr-time choose the inlet noise (npde/stochastic.py).
Writes <out_root>/<name>__spde<tag>/<subject>/spde_<model>.npz and
<out_root>/spde<tag>/{per_roi.csv, summary.json}.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm, wilcoxon  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import list_patients, subject_id  # noqa: E402
from hemopic_plus.npde import transport as TP  # noqa: E402
from hemopic_plus.npde.stochastic import LOG_SIGMA_BOUNDS, StochasticModel  # noqa: E402
from hemopic_plus.npde.uqbatch import load_npde_subject  # noqa: E402
from hemopic_plus.uq import laplace as LP  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE, objective  # noqa: E402

LEVELS = (0.5, 0.8, 0.95)
LOG_SU_OFF = -40.0


def kappa_at_fit(b, theta):
    with torch.no_grad():
        _, _, num, den = objective(b, theta)
    return (LP.n_eff(b) * den / (2.0 * num.clamp(min=1e-30))).numpy()


def fit(m, b, i, theta0, ls0, obs, kappa, stochastic, free_theta=True, maxiter=300):
    """MAP of (theta, log sigma_u, log sigma_v) of ROI i on the frames obs; with
    free_theta False only the noise levels are fitted, theta stays theta0."""
    lo, hi = b.lo[i].numpy(), b.hi[i].numpy()
    theta0 = np.clip(theta0, lo, hi)
    P = theta0.shape[0] if free_theta else 0
    x0 = np.r_[theta0[:P], ls0 if stochastic else ls0[1:]]
    bnds = list(zip(lo, hi))[:P] + [LOG_SIGMA_BOUNDS] * (2 if stochastic else 1)
    obs_t = torch.as_tensor(obs)
    th_fixed = torch.as_tensor(theta0[None, :], dtype=DTYPE)

    def f(x):
        xt = torch.as_tensor(x, dtype=DTYPE).requires_grad_(True)
        th = xt[:P][None, :] if free_theta else th_fixed
        ls_u = xt[P:P + 1] if stochastic else torch.full((1,), LOG_SU_OFF, dtype=DTYPE)
        ls_v = xt[-1:]
        ll = m.loglik(th, ls_u, ls_v, [i], obs_t)[0]
        pr = TP.priors(b.cfg, th, b.log_f0[i:i + 1], b.log_v0[i:i + 1], b.log_cin0[i:i + 1])[0]
        val = -ll + float(kappa) * pr
        val.backward()
        return float(val.detach()), xt.grad.detach().numpy().astype(np.float64)

    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": maxiter})
    th = res.x[:P] if free_theta else theta0
    ls = np.r_[res.x[P], res.x[-1]] if stochastic else np.r_[LOG_SU_OFF, res.x[-1]]
    with torch.no_grad():
        ll = float(m.loglik(torch.as_tensor(th[None, :]), torch.as_tensor(ls[:1]), torch.as_tensor(ls[1:]), [i], obs_t)[0])
    return th, ls, ll


def predictive(m, i, th, ls, obs):
    with torch.no_grad():
        mean, cov, mu = m.predict(torch.as_tensor(th[None, :]), torch.as_tensor(ls[:1]), torch.as_tensor(ls[1:]), [i],
                                  torch.as_tensor(obs))
    return mean[0].numpy(), np.sqrt(np.clip(np.diagonal(cov[0].numpy()), 1e-30, None)), mu[0].numpy()


def load_model(cfg, name, pid, kind, noise, corr_time):
    out, sub = cfg.resolve("out_root"), subject_id(pid)
    fit_dir = out / name / sub
    if kind == "tank":
        b = load_subject(fit_dir)
        return b, StochasticModel(b, n_cells=1, noise=noise, corr_time=corr_time), b.theta_fit.numpy()
    nd = out / f"{name}__npde" / sub
    if not (nd / "npde_theta.npz").exists():
        return None
    pb = load_npde_subject(fit_dir, nd)
    return pb, StochasticModel(pb, noise=noise, corr_time=corr_time), pb.theta_fit.numpy()


def work(args):
    pid, a = args
    torch.set_num_threads(1)
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out, name, sub = cfg.resolve("out_root"), cfg.name, subject_id(pid)
    if not (out / name / sub / "fit_curves.npz").exists():
        return {"patient": pid, "skipped": "no fit"}
    t0 = time.time()
    rows = []
    for kind in a.models:
        got = load_model(cfg, name, pid, kind, a.noise, a.corr_time)
        if got is None:
            continue
        b, m, theta = got
        kap = kappa_at_fit(b, torch.as_tensor(theta, dtype=DTYPE))
        T = b.c_obs.shape[1]
        y = b.c_obs.numpy()
        w = b.w.numpy()
        full = np.ones(T, bool)
        edges = np.linspace(0, T, a.folds + 1).astype(int)
        store = {k: [] for k in ("theta_det", "theta_spde", "ls_det", "ls_spde")}
        for i in range(len(b.roi_ids)):
            ls0 = np.log([0.05, 0.01])
            res = {}
            for ver, sto in (("det", False), ("spde", True)):
                th0 = theta[i] if (ver == "det" or a.joint) else res["det"][0]
                th, ls, ll = fit(m, b, i, th0, ls0, full, kap[i], sto, free_theta=ver == "det" or a.joint)
                _, _, mu = predictive(m, i, th, ls, full)
                nrmse = float(np.sqrt((w[i] * (mu - y[i]) ** 2).mean() / max((w[i] * y[i] ** 2).mean(), 1e-30)))
                res[ver] = (th, ls, ll, nrmse)
                store[f"theta_{ver}"].append(th)
                store[f"ls_{ver}"].append(ls)
            row = {"model": kind, "patient": pid, "roi_id": int(b.roi_ids[i]), "tissue": b.tissue[i],
                   "loglik_gain": res["spde"][2] - res["det"][2],
                   "sigma_u": float(np.exp(res["spde"][1][0])), "sigma_v_spde": float(np.exp(res["spde"][1][1])),
                   "sigma_v_det": float(np.exp(res["det"][1][1])),
                   "nrmse_det": res["det"][3], "nrmse_spde": res["spde"][3]}
            fold_det = {}
            for ver, sto in (("det", False), ("spde", True)):
                hits = {lev: [] for lev in LEVELS}
                lps, wid = [], []
                for f in range(a.folds):
                    hide = np.zeros(T, bool)
                    hide[edges[f]:edges[f + 1]] = True
                    th0 = res[ver][0] if (ver == "det" or a.joint) else fold_det[f]
                    th, ls, _ = fit(m, b, i, th0, res[ver][1], ~hide, kap[i], sto, free_theta=ver == "det" or a.joint)
                    if ver == "det":
                        fold_det[f] = th
                    mean, sd, _ = predictive(m, i, th, ls, ~hide)
                    z = (y[i, hide] - mean[hide]) / sd[hide]
                    for lev in LEVELS:
                        hits[lev].append(np.abs(z) <= norm.ppf(0.5 + lev / 2))
                    lps.append(norm.logpdf(y[i, hide], mean[hide], sd[hide]) + np.log(float(m.y_scale[i])))
                    wid.append(2 * norm.ppf(0.975) * sd[hide] / float(m.y_scale[i]))
                for lev in LEVELS:
                    row[f"cover{int(lev * 100)}_{ver}"] = float(np.mean(np.concatenate(hits[lev])))
                row[f"logscore_{ver}"] = float(np.median(np.concatenate(lps)))
                row[f"width95_{ver}"] = float(np.mean(np.concatenate(wid)))
            rows.append(row)
        d = out / f"{name}__spde{a.tag}" / sub
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"spde_{kind}.npz", roi_ids=b.roi_ids, **{k: np.array(v) for k, v in store.items()})
    return {"patient": pid, "rows": rows, "seconds": round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--models", nargs="*", default=["tank", "npde"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--noise", choices=["relative", "additive"], default="relative")
    p.add_argument("--corr-time", type=float, default=0.0, help="correlation time of the inlet noise in seconds (0: white)")
    p.add_argument("--tag", default="")
    p.add_argument("--joint", action="store_true", help="fit theta together with the noise levels (default: theta from the deterministic fit)")
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out = cfg.resolve("out_root")
    pids = sorted(a.patients or list_patients(cfg.resolve("data_root")))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, [(pid, a) for pid in pids]):
            if "skipped" in r:
                continue
            rows += r["rows"]
            for kind in a.models:
                rr = [x for x in r["rows"] if x["model"] == kind]
                if rr:
                    print(f"  training_{r['patient']:<3d} {kind:4s} {r['seconds']:6.0f}s  dloglik {np.median([x['loglik_gain'] for x in rr]):7.2f}  "
                          f"cover95 det {np.mean([x['cover95_det'] for x in rr]):.3f} spde {np.mean([x['cover95_spde'] for x in rr]):.3f}", flush=True)
    od = out / f"spde{a.tag}"
    od.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(od / "per_roi.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    summ = {}
    for kind in a.models:
        rr = [x for x in rows if x["model"] == kind]
        if not rr:
            continue
        s = {"n_roi": len(rr), "n_patients": len({x["patient"] for x in rr}),
             "loglik gain, median": float(np.median([x["loglik_gain"] for x in rr])),
             "fraction with gain > 2": float(np.mean([x["loglik_gain"] > 2 for x in rr])),
             "sigma_u, median": float(np.median([x["sigma_u"] for x in rr]))}
        for ver in ("det", "spde"):
            for k in ("cover50", "cover80", "cover95", "logscore", "width95", "nrmse"):
                key = f"{k}_{ver}"
                s[key] = float(np.mean([x[key] for x in rr]))
        for k in ("cover95", "logscore", "nrmse"):
            pp = {}
            for x in rr:
                pp.setdefault(x["patient"], []).append(x[f"{k}_spde"] - x[f"{k}_det"])
            d = np.array([np.mean(v) for v in pp.values()])
            s[f"{k} spde - det, per patient"] = {"diff": float(d.mean()), "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
        summ[kind] = s
    (od / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
