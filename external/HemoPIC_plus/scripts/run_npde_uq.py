"""Uncertainty of the neural PDE (run after train_npde.py, with --extra-members
for the network ensemble).

python scripts/run_npde_uq.py configs/full_aif.yaml [--workers 30]

For every subject, with the closure of its held-out fold frozen:
  1. posterior of the eight PDE parameters per ROI (same tempered posterior,
     Laplace and MCMC as run_uq.py) and voxel sd / 95% interval maps;
  2. Check 1, hidden time frames (as eval_heldout.py);
  3. Check 3, semi-synthetic: the neural PDE fit is the truth, noise with the
     level and time correlation of the real residuals is added, the curve is
     refitted from the priors with the PDE estimator, and the 95% interval of
     CBF, V, MTT and tau is checked against the truth;
  4. network uncertainty: spread of the ROI quantities and of the maps across the
     bootstrap networks (members) of the fold, and lesion AUC per member.
Writes <out_root>/<name>__npde/training_XX/{uq,ens}/ and
       <out_root>/<name>__npde/eval_uq/{heldout.csv, semisynth.csv, ensemble.json, members.csv}.
Then run eval_uq.py with --name <name>__npde for the cohort summary.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from eval_heldout import LEVELS, heldout_rows  # noqa: E402
from run_pde import rescaled_maps  # noqa: E402
from run_uq import posterior_for  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, subject_id  # noqa: E402
from hemopic_plus.npde.uqbatch import load_npde_subject  # noqa: E402
from hemopic_plus.uq import laplace as LP  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE, derived, objective, simulate  # noqa: E402

QTY = ("CBF", "V", "MTT", "tau")
PE_STARTS = (0.3, 3.0, 30.0)


def lbfgs(pb, theta0, maxiter=300):
    B, P = theta0.shape

    def f(x):
        th = torch.as_tensor(x, dtype=DTYPE).reshape(B, P).requires_grad_(True)
        val = objective(pb, th)[0].sum()
        val.backward()
        return float(val.detach()), th.grad.detach().numpy().ravel().astype(np.float64)

    res = minimize(f, theta0.numpy().ravel(), jac=True, method="L-BFGS-B",
                   bounds=list(zip(pb.lo.numpy().ravel(), pb.hi.numpy().ravel())), options={"maxiter": maxiter})
    th = torch.as_tensor(res.x, dtype=DTYPE).reshape(B, P)
    with torch.no_grad():
        return th, objective(pb, th)[0]


def prior_start(pb):
    """The pipeline's starting point (fit_roi_curve), plus a Pe start."""
    cfg, t, dt = pb.cfg, pb.t.numpy(), pb.dt
    t_end = float(t[-1])
    tau_min, tau_max = max(dt, 0.001), min(float(cfg.tau_cap_seconds), 0.98 * t_end)
    tau0 = max(min(0.3 * t_end, tau_max), tau_min)
    B = len(pb.roi_ids)
    th = torch.zeros(B, 8, dtype=DTYPE)
    th[:, 1] = float(np.log(tau0))
    th[:, 2], th[:, 3], th[:, 4] = pb.log_v0, pb.log_f0, pb.log_cin0
    th[:, 6] = float(np.log(max(cfg.dispersion_bounds[0], 1e-3)))
    return th


def semisynth_rows(pb, name, reps, seed):
    theta_true = LP.refine_map(pb)
    with torch.no_grad():
        _, c_true = simulate(pb, theta_true)
        _, _, num, _ = objective(pb, theta_true)
        truth = {k: v.numpy() for k, v in derived(pb, theta_true).items()}
    rng = np.random.default_rng(seed)
    T = c_true.shape[1]
    w = pb.w[0].numpy()
    B = len(pb.roi_ids)
    rows = []
    for rep in range(reps):
        c_noisy = c_true.numpy().copy()
        for i in range(B):
            e = gaussian_filter1d(rng.standard_normal(T), sigma=float(pb.cfg.smooth_sigma_frames), mode="nearest")
            e *= np.sqrt(float(num[i]) / max(np.mean(w * e ** 2), 1e-30))
            c_noisy[i] += e
        pn = pb.subset(range(B))
        pn.c_obs = torch.as_tensor(c_noisy, dtype=DTYPE)
        best, best_obj = None, None
        for pe in PE_STARTS:
            th0 = prior_start(pn)
            th0[:, 7] = float(np.log(pe))
            th0 = torch.minimum(torch.maximum(th0, pn.lo), pn.hi)
            th, ob = lbfgs(pn, th0)
            if best is None:
                best, best_obj = th.clone(), ob.clone()
            else:
                better = ob < best_obj
                best[better], best_obj[better] = th[better], ob[better]
        pn.theta_fit = best
        theta_hat = LP.refine_map(pn)
        kappa = LP.temperature(pn, theta_hat)
        cov, _, _ = LP.laplace(pn, theta_hat, kappa)
        draws = LP.sample_gaussian(pn, theta_hat, cov, n=1000, seed=rep)
        d = {k: v.numpy() for k, v in LP.derived_samples(pn, draws).items()}
        for i in range(B):
            row = {"subject": name, "roi_id": int(pb.roi_ids[i]), "tissue": pb.tissue[i], "rep": rep}
            for k in QTY:
                lo, hi = np.percentile(d[k][i], [2.5, 97.5])
                row[f"{k}_true"] = float(truth[k][i])
                row[f"{k}_covered"] = bool(lo <= truth[k][i] <= hi)
                row[f"{k}_rel_width"] = float((hi - lo) / max(truth[k][i], 1e-12))
                row[f"{k}_rel_err"] = float(np.median(d[k][i]) / max(truth[k][i], 1e-12) - 1.0)
            rows.append(row)
    return rows


def member_spread(fit_dir, nd, data_root, pb):
    """Network uncertainty from the bootstrap members of the fold."""
    z = np.load(nd / "npde_theta.npz")
    if "theta_members" not in z:
        return None, []
    M = z["theta_members"].shape[0]
    with torch.no_grad():
        d_main = {k: v.numpy() for k, v in derived(pb, pb.theta_fit).items()}
        d_m = [{k: v.numpy() for k, v in derived(pb, torch.as_tensor(z["theta_members"][m], dtype=DTYPE)).items()}
               for m in range(M)]
    sub = nd.name
    seg = nib.load(str(data_root / sub / SEG4_REL)).get_fdata() > 0
    ot = nib.load(str(find_nii(data_root / sub, "OT"))).get_fdata() > 0
    vols = {m: {} for m, _ in MAPS}
    auc = {m: [] for m, _ in MAPS}
    for k in range(M):
        img, cbf, cbv, mtt = rescaled_maps(nd, pb.roi_ids, d_m[k]["CBF"] / d_main["CBF"], d_m[k]["V"] / d_main["V"], pb.cfg)
        for (m, direction), v in zip(MAPS, (cbf, cbv, mtt)):
            vols[m][k] = v
            valid, score = _lesion_scores(v, seg, direction)
            auc[m].append(roc_auc(ot[valid], score) if score is not None else np.nan)
    (nd / "ens").mkdir(exist_ok=True)
    rel_sd, rel_post = {}, {}
    for m, _ in MAPS:
        stack = np.stack(list(vols[m].values()))
        mean, sd = np.nanmean(stack, 0), np.nanstd(stack, 0)
        nib.save(nib.Nifti1Image(sd.astype(np.float32), img.affine), str(nd / "ens" / f"HemoPIC_{m}_ens_sd.nii.gz"))
        ok = seg & np.isfinite(mean) & (mean > 0)
        rel_sd[m] = float(np.nanmedian(sd[ok] / mean[ok]))
        ps = nib.load(str(nd / "uq" / f"HemoPIC_{m}_sd.nii.gz")).get_fdata()
        v0 = vols[m][0]
        okp = ok & np.isfinite(ps) & (v0 > 0)
        rel_post[m] = float(np.nanmedian(ps[okp] / v0[okp]))
    rows = []
    for i, rid in enumerate(pb.roi_ids):
        r = {"subject": sub, "roi_id": int(rid), "tissue": pb.tissue[i]}
        for q in QTY:
            x = np.log([d[q][i] for d in d_m])
            r[f"{q}_member_logsd"] = float(np.std(x))
        r["nu_fast_sd"] = float(np.std([np.max(z["phi_members"][m, i] / z["nu_members"][m, i]) for m in range(M)]))
        rows.append(r)
    return {"auc": auc, "rel_sd_members": rel_sd, "rel_sd_posterior": rel_post, "M": M}, rows


def run_subject(args):
    fit_dir, nd, data_root, burn, keep, chains, folds, draws, reps = args
    torch.set_num_threads(1)
    t0 = time.time()
    pb = load_npde_subject(fit_dir, nd)
    pid = int(nd.name.split("_")[1])
    info = posterior_for(pb, nd, burn, keep, chains, pid)
    hrows = heldout_rows(pb, nd.name, folds, draws)
    srows = semisynth_rows(pb, nd.name, reps, pid)
    ens, mrows = member_spread(fit_dir, nd, data_root, pb)
    zp = np.load(nd / "uq" / "posterior.npz")
    for r, i in zip(mrows, range(len(mrows))):
        for q in QTY:
            r[f"{q}_post_logsd"] = float(np.std(np.log(zp[q][i])))
    return nd.name, info, hrows, srows, ens, mrows, time.time() - t0


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--burn", type=int, default=3000)
    p.add_argument("--keep", type=int, default=6000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--draws", type=int, default=400)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    root = cfg.resolve("out_root") / cfg.name
    npde_root = root.parent / f"{cfg.name}__npde"
    data_root = cfg.resolve("data_root")
    pids = a.patients or list_patients(npde_root)
    jobs = [(root / subject_id(pid), npde_root / subject_id(pid), data_root, a.burn, a.keep, a.chains, a.folds, a.draws, a.reps)
            for pid in pids if (npde_root / subject_id(pid) / "npde_theta.npz").exists()]
    workers = a.workers or os.cpu_count() or 1
    print(f"[{cfg.name}__npde] uncertainty on {len(jobs)} subjects, {workers} workers")
    t0 = time.time()
    H, S, E, MR = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, info, hrows, srows, ens, mrows, sec in ex.map(run_subject, jobs):
            H += hrows
            S += srows
            MR += mrows
            if ens:
                E.append(ens)
            if info["rhat_max"] > 2:
                print(f"  {name}: R-hat max {info['rhat_max']:.2f} ({100 * info['frac_rhat_gt_1_1']:.0f}% ROIs > 1.1)", flush=True)
    print(f"[{cfg.name}__npde] done in {time.time() - t0:.0f}s")
    out = npde_root / "eval_uq"
    out.mkdir(exist_ok=True)
    write_csv(out / "heldout.csv", H)
    write_csv(out / "semisynth.csv", S)
    print("  check 1, hidden frames (weighted by hidden frames):")
    for lev in LEVELS:
        sel = [r for r in H if r["level"] == lev]
        n = np.array([r["n"] for r in sel])
        cov = np.sum(n * np.array([r["covered"] for r in sel])) / n.sum()
        cov_n = np.sum(n * np.array([r["covered_noise_only"] for r in sel])) / n.sum()
        print(f"    nominal {lev:.0%}: posterior predictive {cov:.1%}   noise only {cov_n:.1%}")
    print(f"  check 3, semi-synthetic coverage of the 95% interval ({len(S)} ROI fits):")
    for k in QTY:
        print(f"    {k:4s}: coverage {np.mean([r[f'{k}_covered'] for r in S]):.1%}   median width "
              f"{np.median([r[f'{k}_rel_width'] for r in S]):.0%} of truth   median |error| {np.median(np.abs([r[f'{k}_rel_err'] for r in S])):.1%}")
    if E:
        write_csv(out / "members.csv", MR)
        M = E[0]["M"]
        summ = {"members": M,
                "auc_per_member": {m: [float(np.nanmean([e["auc"][m][k] for e in E])) for k in range(M)] for m, _ in MAPS},
                "median_rel_sd_members": {m: float(np.median([e["rel_sd_members"][m] for e in E])) for m, _ in MAPS},
                "median_rel_sd_posterior": {m: float(np.median([e["rel_sd_posterior"][m] for e in E])) for m, _ in MAPS},
                "roi_logsd_members_median": {q: float(np.median([r[f"{q}_member_logsd"] for r in MR])) for q in QTY},
                "roi_logsd_posterior_median": {q: float(np.median([r[f"{q}_post_logsd"] for r in MR])) for q in QTY}}
        (out / "ensemble.json").write_text(json.dumps(summ, indent=2), encoding="utf8")
        print(f"  network ensemble ({M} networks per fold): lesion AUC per member (CBF / CBV / MTT):")
        for k in range(M):
            print("    member %d  %s" % (k, " / ".join(f"{summ['auc_per_member'][m][k]:.4f}" for m, _ in MAPS)))
        print("    voxel relative sd, networks vs posterior: " + "  ".join(
            f"{m} {summ['median_rel_sd_members'][m]:.1%} vs {summ['median_rel_sd_posterior'][m]:.1%}" for m, _ in MAPS))
        print("    ROI sd of log quantity, networks vs posterior: " + "  ".join(
            f"{q} {summ['roi_logsd_members_median'][q]:.3f} vs {summ['roi_logsd_posterior_median'][q]:.3f}" for q in QTY))


if __name__ == "__main__":
    main()
