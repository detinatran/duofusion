"""Check 3 (light semi-synthetic): do the intervals contain a known answer?

python scripts/eval_semisynth.py configs/full.yaml [--reps 3]

For every ROI the fitted curve (posterior mode of the real data) is taken as
the truth. Noise with the same weighted variance as the real residuals and the
same temporal correlation (white noise smoothed with the pipeline's Gaussian,
sigma = smooth_sigma_frames) is added, the curve is refitted with the
pipeline's own estimator (fit_roi_curve, then L-BFGS-B polish as in run_uq.py),
and the Laplace 95% interval of CBF level, V, MTT and tau is checked against the
known truth. Coverage near 95% means the intervals can be trusted.

Writes <out_root>/<name>/eval_uq/semisynth.csv and prints coverage per quantity.
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
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.fit import fit_roi_curve  # noqa: E402
from hemopic_plus.io import list_patients, subject_id  # noqa: E402
from hemopic_plus.uq import laplace as LP  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE, derived, objective, simulate  # noqa: E402

QTY = ("CBF", "V", "MTT", "tau")


def run_subject(args):
    fit_dir, reps = args
    torch.set_num_threads(1)
    b = load_subject(fit_dir)
    cfg = b.cfg
    theta_true = LP.refine_map(b)
    with torch.no_grad():
        _, c_true = simulate(b, theta_true)
        _, _, num, _ = objective(b, theta_true)
        truth = {k: v.numpy() for k, v in derived(b, theta_true).items()}
    rng = np.random.default_rng(int(fit_dir.name.split("_")[1]))
    t, dt = b.t.numpy(), b.dt
    shape_np = b.shape.numpy()
    w = b.w[0].numpy()
    rows = []
    for rep in range(reps):
        c_noisy = np.empty_like(c_true.numpy())
        for i in range(len(b.roi_ids)):
            e = gaussian_filter1d(rng.standard_normal(t.size), sigma=float(cfg.smooth_sigma_frames), mode="nearest")
            e *= np.sqrt(float(num[i]) / max(np.mean(w * e ** 2), 1e-30))
            c_noisy[i] = c_true[i].numpy() + e
        bn = b.subset(range(len(b.roi_ids)))
        bn.c_obs = torch.as_tensor(c_noisy, dtype=DTYPE)
        refit = []
        for i in range(len(b.roi_ids)):
            fr = fit_roi_curve(c_noisy[i], shape_np, t, dt, float(torch.exp(b.log_f0[i])), float(torch.exp(b.log_v0[i])),
                               float(torch.exp(b.log_cin0[i])), cfg, rng=np.random.default_rng(rep), w_curve=b.w_curve_raw)
            th = [np.log(fr["R"]), np.log(fr["tau"]), np.log(fr["V"]), np.log(fr["F"]), np.log(fr["Cin_bar"])]
            if cfg.delay_dispersion:
                th += [fr["delay"], np.log(max(fr["dispersion"], 1e-12))]
            refit.append(th)
        bn.theta_fit = torch.as_tensor(np.array(refit), dtype=DTYPE)
        bn.w = b.w
        theta_hat = LP.refine_map(bn)
        kappa = LP.temperature(bn, theta_hat)
        cov, pinned, _ = LP.laplace(bn, theta_hat, kappa)
        draws = LP.sample_gaussian(bn, theta_hat, cov, n=1000, seed=rep)
        d = {k: v.numpy() for k, v in LP.derived_samples(bn, draws).items()}
        for i in range(len(b.roi_ids)):
            row = {"subject": fit_dir.name, "roi_id": int(b.roi_ids[i]), "tissue": b.tissue[i], "rep": rep}
            for k in QTY:
                lo, hi = np.percentile(d[k][i], [2.5, 97.5])
                row[f"{k}_true"] = float(truth[k][i])
                row[f"{k}_covered"] = bool(lo <= truth[k][i] <= hi)
                row[f"{k}_rel_width"] = float((hi - lo) / max(truth[k][i], 1e-12))
                row[f"{k}_rel_err"] = float(np.median(d[k][i]) / max(truth[k][i], 1e-12) - 1.0)
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("out_root", a.out_root),) if v})
    root = cfg.resolve("out_root") / cfg.name
    pids = a.patients or list_patients(root)
    jobs = [(root / subject_id(pid), a.reps) for pid in pids if (root / subject_id(pid) / "fit_curves.npz").exists()]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers or os.cpu_count() or 1) as ex:
        for r in ex.map(run_subject, jobs):
            rows.extend(r)
    out = root / "eval_uq"
    out.mkdir(exist_ok=True)
    with open(out / "semisynth.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[{cfg.name}] semi-synthetic coverage of the 95% interval ({len(rows)} ROI fits, {len(jobs)} subjects)")
    for k in QTY:
        cov = np.mean([r[f"{k}_covered"] for r in rows])
        wid = np.median([r[f"{k}_rel_width"] for r in rows])
        err = np.median(np.abs([r[f"{k}_rel_err"] for r in rows]))
        print(f"  {k:4s}: coverage {cov:.1%}   median interval width {wid:.0%} of truth   median |error| {err:.1%}")


if __name__ == "__main__":
    main()
