"""Neural operator as the solver of the transport PDE (a surrogate), against the
exact modal solver it learns from.

python scripts/bench_operator_solver.py configs/full_aif.yaml [--test-fold 0] [--samples 40] [--steps 4000] [--threads 30]

The physics is kept: the input is the arterial curve, the eight PDE parameters
and the two-tube split (nu, phi) of the closure; the output is the tissue curve
the exact solver gives for them. Training curves: the neural PDE fit of every
ROI of the training patients (folds as train_npde.py) and --samples random
perturbations around it (log parameters +-0.3, delay +-1 s, log Pe +-0.7, split
+-0.3 in logit), simulated with the exact solver. The target is divided by
Cin (the peak of the tissue curve is close to Cin), so it is of order one. Reported on the held-out patients:
  1. surrogate error, weighted normalised squared error against the exact solver,
     at the fitted parameters, at perturbed ones and out of range (MTT x3,
     Pe at its bounds), next to the model error (exact fit against the data);
  2. time per curve, forward and forward + gradient, same threads;
  3. the fit with the surrogate in place of the solver (same objective, priors
     and optimiser, from the same start as an exact refit): change of CBF, V and
     MTT against the exact refit, and the true data error (exact solver) at the
     surrogate's parameters.
Writes <out_root>/<name>__operator/solver_bench.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from train_operator import FNO  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import list_patients, subject_id  # noqa: E402
from hemopic_plus.npde import neural as NN  # noqa: E402
from hemopic_plus.npde import transport as TP  # noqa: E402

PERTURB_SD = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 0.3, 0.7])


def logit(p):
    return torch.log(p / (1 - p))


class Surrogate(torch.nn.Module):
    def __init__(self, mu, sd, width, modes, layers):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.fno = FNO(3 + 8 + 3, width, modes, layers)

    def forward(self, co, theta, nu1, phi1):
        T = co.c_obs.shape[1]
        shape = co.shape / co.shape.max(dim=1, keepdim=True).values.clamp(min=1e-12)
        t = (co.t - co.t[:, :1]) / 100.0
        mask = (co.w > 0).to(shape.dtype)
        static = torch.cat([(theta - self.mu) / self.sd, logit(nu1)[:, None], logit(phi1)[:, None], co.dt[:, None]], 1)
        x = torch.cat([shape[:, None], t[:, None], mask[:, None], static[:, :, None].expand(-1, -1, T)], 1).float()
        y = self.fno(x).double()
        return y * torch.exp(theta[:, 4])[:, None]


def exact(co, theta, nu1, phi1, n_cells):
    paths = (torch.stack([nu1, 1 - nu1], 1), torch.stack([phi1, 1 - phi1], 1))
    return NN.simulate(co, theta, n_cells, paths)[1]


def nerr(co, a, b):
    return ((co.w * (a - b) ** 2).sum(1) / ((co.w * b ** 2).sum(1) + 1e-30))


def energy_ok(co, y, theta, floor=1e-6):
    """Curve energy relative to Cin squared, per frame."""
    e = (co.w * (y / torch.exp(theta[:, 4])[:, None]) ** 2).sum(1) / co.w.sum(1).clamp(min=1e-12)
    return e > floor * float(e.median())


def perturb(co, theta, nu1, phi1, g):
    th = theta + PERTURB_SD.to(theta.dtype) * torch.randn(theta.shape, generator=g, dtype=theta.dtype)
    th = torch.minimum(torch.maximum(th, co.lo + 1e-6), co.hi - 1e-6)
    j = lambda p: torch.sigmoid(logit(p) + 0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype)).clamp(0.05, 0.95)
    return th, j(nu1), j(phi1)


def refit(co, theta0, nu1, phi1, sim, steps, lr=0.02):
    x = NN.from_theta(co, theta0).clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        th = NN.to_theta(co, x)
        c = sim(co, th, nu1, phi1)
        data = (co.w * (c - co.c_obs) ** 2).sum(1) / ((co.w * co.c_obs ** 2).sum(1) + 1e-12)
        loss = (data + TP.priors(co.cfg, th, co.log_f0, co.log_v0, co.log_cin0)).mean()
        loss.backward()
        opt.step()
        sched.step()
    return NN.to_theta(co, x).detach()


def timed(fn, reps=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def q(x):
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(np.max(x))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--out-root")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test-fold", type=int, default=0)
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--refit-steps", type=int, default=800)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--modes", type=int, default=24)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--threads", type=int, default=30)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_config(a.config, {k: v for k, v in (("out_root", a.out_root),) if v})
    root = cfg.resolve("out_root") / cfg.name
    npde_root = root.parent / f"{cfg.name}__npde"
    pids = sorted(pid for pid in list_patients(root) if (root / subject_id(pid) / "fit_curves.npz").exists())
    rng = np.random.default_rng(0)
    fold_of = {pid: f for pid, f in zip(rng.permutation(pids), np.arange(len(pids)) % a.folds)}
    pids = [pid for pid in pids if (npde_root / subject_id(pid) / "npde_theta.npz").exists()]
    co = NN.load_cohort([root / subject_id(pid) for pid in pids])
    TH, NU, PHI = [], [], []
    for pid in pids:
        z = np.load(npde_root / subject_id(pid) / "npde_theta.npz")
        TH.append(z["theta"])
        NU.append(z["nu"][:, 0])
        PHI.append(z["phi"][:, 0])
    theta = torch.as_tensor(np.concatenate(TH), dtype=torch.float64)
    nu1 = torch.as_tensor(np.concatenate(NU), dtype=torch.float64)
    phi1 = torch.as_tensor(np.concatenate(PHI), dtype=torch.float64)
    theta = torch.minimum(torch.maximum(theta, co.lo + 1e-6), co.hi - 1e-6)
    n_cells = 24
    roi_fold = np.array([fold_of[s] for s in co.subject])
    tr_idx, te_idx = np.where(roi_fold != a.test_fold)[0], np.where(roi_fold == a.test_fold)[0]
    print(f"[{cfg.name}] solver surrogate: {len(pids)} patients, {len(tr_idx)} train / {len(te_idx)} test ROIs", flush=True)

    g = torch.Generator().manual_seed(0)
    rep = np.repeat(tr_idx, a.samples + 1)
    ct = co.take(rep)
    ti = torch.as_tensor(rep)
    th_t, nu_t, phi_t = theta[ti].clone(), nu1[ti].clone(), phi1[ti].clone()
    pert = torch.as_tensor(np.tile(np.r_[False, np.ones(a.samples, bool)], len(tr_idx)))
    th_p, nu_p, phi_p = perturb(ct, th_t, nu_t, phi_t, g)
    th_t[pert], nu_t[pert], phi_t[pert] = th_p[pert], nu_p[pert], phi_p[pert]
    t0 = time.time()
    with torch.no_grad():
        y_t = torch.cat([exact(ct.take(np.arange(i, min(i + 2048, len(rep)))), th_t[i:i + 2048], nu_t[i:i + 2048],
                               phi_t[i:i + 2048], n_cells) for i in range(0, len(rep), 2048)])
    print(f"  {len(rep)} training curves simulated in {time.time() - t0:.0f}s", flush=True)
    keep = energy_ok(ct, y_t, th_t)
    print(f"  {int((~keep).sum())} near-empty curves left out", flush=True)
    kept = torch.where(keep)[0]
    ct, th_t, nu_t, phi_t, y_t = ct.take(kept.numpy()), th_t[kept], nu_t[kept], phi_t[kept], y_t[kept]
    rep = rep[kept.numpy()]

    mu, sd = th_t.mean(0), th_t.std(0) + 1e-6
    net = Surrogate(mu, sd, a.width, a.modes, a.layers)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    t0 = time.time()
    for step in range(a.steps):
        b = torch.randint(0, len(rep), (a.batch,), generator=g)
        cb = ct.take(b.numpy())
        loss = nerr(cb, net(cb, th_t[b], nu_t[b], phi_t[b]), y_t[b]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"    step {step:5d} train error {float(loss.detach()):.5f} ({time.time() - t0:.0f}s)", flush=True)
    for prm in net.parameters():
        prm.requires_grad_(False)

    ce = co.take(te_idx)
    ei = torch.as_tensor(te_idx)
    th_e, nu_e, phi_e = theta[ei], nu1[ei], phi1[ei]
    res = {"n_train_curves": len(rep), "n_test_roi": len(te_idx), "settings": vars(a) | {"config": str(a.config)}}
    cases = {"fitted": (th_e, nu_e, phi_e)}
    g2 = torch.Generator().manual_seed(1)
    cases["perturbed"] = perturb(ce, th_e, nu_e, phi_e, g2)
    th = th_e.clone()
    th[:, 2] = torch.minimum(th[:, 2] + float(np.log(3.0)), ce.hi[:, 2] - 1e-6)
    cases["MTT x3"] = (th, nu_e, phi_e)
    for lab, j in (("Pe at upper bound", 1), ("Pe at lower bound", 0)):
        th = th_e.clone()
        th[:, 7] = (ce.hi[:, 7] - 1e-6) if j else (ce.lo[:, 7] + 1e-6)
        cases[lab] = (th, nu_e, phi_e)
    with torch.no_grad():
        c_fit = exact(ce, th_e, nu_e, phi_e, n_cells)
        res["model error (exact fit vs data)"] = q(nerr(ce, c_fit, ce.c_obs).numpy())
        for lab, (th, nu, ph) in cases.items():
            y = exact(ce, th, nu, ph, n_cells)
            ok = energy_ok(ce, y, th)
            res[f"surrogate error vs exact, {lab}"] = q(nerr(ce, net(ce, th, nu, ph), y)[ok].numpy()) | {"n": int(ok.sum())}

    def fwd(sim):
        with torch.no_grad():
            sim(ce, th_e, nu_e, phi_e)

    def grad(sim):
        x = th_e.clone().requires_grad_(True)
        sim(ce, x, nu_e, phi_e).sum().backward()

    ex = lambda c, t_, n_, p_: exact(c, t_, n_, p_, n_cells)
    B = len(te_idx)
    res["ms per curve, forward, exact"] = 1e3 * timed(lambda: fwd(ex)) / B
    res["ms per curve, forward, surrogate"] = 1e3 * timed(lambda: fwd(net)) / B
    res["ms per curve, forward + gradient, exact"] = 1e3 * timed(lambda: grad(ex)) / B
    res["ms per curve, forward + gradient, surrogate"] = 1e3 * timed(lambda: grad(net)) / B

    t0 = time.time()
    th_ex = refit(ce, th_e, nu_e, phi_e, ex, a.refit_steps)
    th_su = refit(ce, th_e, nu_e, phi_e, net, a.refit_steps)
    print(f"  refits done in {time.time() - t0:.0f}s", flush=True)
    F_ex, F_su = torch.exp(th_ex[:, 3]), torch.exp(th_su[:, 3])
    V_ex, V_su = torch.exp(th_ex[:, 2]), torch.exp(th_su[:, 2])
    for lab, x_ex, x_su in (("CBF (F)", F_ex, F_su), ("V", V_ex, V_su), ("MTT", V_ex / F_ex, V_su / F_su)):
        res[f"|relative change| surrogate fit vs exact fit, {lab}"] = q((x_su / x_ex - 1).abs().numpy())
    with torch.no_grad():
        res["data error at exact refit (exact solver)"] = q(nerr(ce, exact(ce, th_ex, nu_e, phi_e, n_cells), ce.c_obs).numpy())
        res["data error at surrogate refit (exact solver)"] = q(nerr(ce, exact(ce, th_su, nu_e, phi_e, n_cells), ce.c_obs).numpy())
        res["data error at surrogate refit (surrogate)"] = q(nerr(ce, net(ce, th_su, nu_e, phi_e), ce.c_obs).numpy())
    od = root.parent / f"{cfg.name}__operator"
    od.mkdir(parents=True, exist_ok=True)
    (od / "solver_bench.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
