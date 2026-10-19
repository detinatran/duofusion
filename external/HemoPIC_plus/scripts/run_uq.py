"""Uncertainty of a fitted experiment: Laplace + MCMC posterior per ROI, and
voxel-wise uncertainty maps.

python scripts/run_uq.py configs/full.yaml                     # every fitted subject
python scripts/run_uq.py configs/full.yaml --patients 22 --workers 1

Reads  <out_root>/<name>/training_XX/ (run_cohort.py output)
Writes <out_root>/<name>/training_XX/uq/
  roi_posterior.csv     per ROI: posterior mean/sd/95% CI of CBF, V, MTT, tau, F,
                        posterior/prior sd ratio per parameter, pinned flags, R-hat, ESS
  posterior.npz         thinned posterior draws (parameters and derived quantities)
  HemoPIC_{CBF,CBV,MTT}_{sd,lo,hi}.nii.gz   voxel-wise posterior sd and 95% interval
  uq_info.json          temperature, acceptance, convergence summary
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

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import list_patients, subject_id  # noqa: E402
from hemopic_plus.uq import laplace as LP  # noqa: E402
from hemopic_plus.uq import mcmc as MC  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import derived  # noqa: E402

MAP_DRAWS = 200


def voxel_maps(fit_dir, batch, d_fit, d_samp, out):
    """Voxel posterior of the maps. Inside ROI r every map is the fitted map scaled by
    level_r(sample) / level_r(fit) (CBF level; V for CBV unless cbv_source is an
    indicator-dilution option); MTT is recomputed with the run's floors and caps."""
    info = batch.info
    cfg = batch.cfg
    ref = nib.load(str(fit_dir / "HemoPIC_CBF.nii.gz"))
    cbf = ref.get_fdata()
    cbv = nib.load(str(fit_dir / "HemoPIC_CBV.nii.gz")).get_fdata()
    lab = nib.load(str(fit_dir / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
    cbf_floor, cbf_cap = info["cbf_floor_from_gt_p1"], info["cbf_cap_from_gt_p99_5"]
    mtt_floor, mtt_cap = info["mtt_floor_from_gt_p1"], info["mtt_cap_from_gt_p99_5"]
    scale_cbv = cfg.cbv_source == "model_v"
    res = {k: {s: np.full(cbf.shape, np.nan, dtype=np.float32) for s in ("sd", "lo", "hi")} for k in ("CBF", "CBV", "MTT")}
    rng = np.random.default_rng(0)
    for i, rid in enumerate(batch.roi_ids):
        m = lab == int(rid)
        if not m.any():
            continue
        S = d_samp["CBF"].shape[1]
        pick = rng.choice(S, size=min(MAP_DRAWS, S), replace=False)
        rf = (d_samp["CBF"][i, pick] / d_fit["CBF"][i]).astype(np.float32)
        rv = (d_samp["V"][i, pick] / d_fit["V"][i]).astype(np.float32) if scale_cbv else np.ones_like(rf)
        f_vox = cbf[m].astype(np.float32)[:, None] * rf[None, :]
        v_vox = cbv[m].astype(np.float32)[:, None] * rv[None, :]
        den = np.clip(f_vox, cbf_floor, cbf_cap)
        mtt = np.clip(60.0 * v_vox / den, mtt_floor, mtt_cap)
        for k, arr in (("CBF", f_vox), ("CBV", v_vox), ("MTT", mtt)):
            res[k]["sd"][m] = arr.std(axis=1)
            res[k]["lo"][m] = np.percentile(arr, 2.5, axis=1)
            res[k]["hi"][m] = np.percentile(arr, 97.5, axis=1)
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.float32)
    for k, d in res.items():
        for s, arr in d.items():
            nib.save(nib.Nifti1Image(arr, ref.affine, hdr), str(out / f"HemoPIC_{k}_{s}.nii.gz"))


def run_subject(args):
    fit_dir, n_burn, n_keep, chains, seed = args
    torch.set_num_threads(1)
    return fit_dir.name, posterior_for(load_subject(fit_dir), fit_dir, n_burn, n_keep, chains, seed)


def posterior_for(b, fit_dir, n_burn, n_keep, chains, seed):
    """Posterior of every ROI of batch b; maps and labels are read from fit_dir,
    results go to fit_dir/uq. Returns the run summary."""
    t0 = time.time()
    out = fit_dir / "uq"
    out.mkdir(exist_ok=True)
    theta = LP.refine_map(b)
    kappa = LP.temperature(b, theta)
    cov, pinned, _ = LP.laplace(b, theta, kappa)
    lap_sd = torch.sqrt(torch.clamp(torch.diagonal(cov, dim1=1, dim2=2), min=0.0))
    psd = LP.prior_sd(b, kappa)
    samples, acc = MC.run(b, theta, cov, pinned, kappa, chains=chains, n_burn=n_burn, n_keep=n_keep, seed=seed)
    rhat, ess = MC.rhat_ess(samples)
    B, C, S, P = samples.shape
    flat = samples.reshape(B, C * S, P)
    post_sd = flat.std(dim=1)
    d_samp = {k: v.numpy() for k, v in LP.derived_samples(b, flat).items()}
    with torch.no_grad():
        d_fit = {k: v.numpy() for k, v in derived(b, b.theta_fit).items()}
        d_map = {k: v.numpy() for k, v in derived(b, theta).items()}

    rows = []
    for i, rid in enumerate(b.roi_ids):
        r = {"roi_id": int(rid), "tissue": b.tissue[i], "kappa": float(kappa[i]), "acceptance": float(acc[i]),
             "rhat_max": float(rhat[i].max()), "ess_min": float(ess[i].min())}
        for k in ("CBF", "V", "MTT", "tau", "F"):
            x = d_samp[k][i]
            r[f"{k}_fit"] = float(d_fit[k][i])
            r[f"{k}_map"] = float(d_map[k][i])
            r[f"{k}_mean"] = float(x.mean())
            r[f"{k}_sd"] = float(x.std())
            r[f"{k}_lo"] = float(np.percentile(x, 2.5))
            r[f"{k}_hi"] = float(np.percentile(x, 97.5))
        for k, name in enumerate(b.names):
            r[f"{name}_post_sd"] = float(post_sd[i, k])
            r[f"{name}_laplace_sd"] = float(lap_sd[i, k])
            r[f"{name}_prior_sd"] = float(psd[i, k])
            r[f"{name}_pinned"] = bool(pinned[i, k])
        rows.append(r)
    with open(out / "roi_posterior.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    keep = np.linspace(0, C * S - 1, min(1000, C * S)).astype(int)
    np.savez_compressed(out / "posterior.npz", roi_ids=b.roi_ids, names=np.array(b.names),
                        theta=flat[:, keep].numpy().astype(np.float32), theta_map=theta.numpy(),
                        **{k: v[:, keep].astype(np.float32) for k, v in d_samp.items()})
    voxel_maps(fit_dir, b, d_fit, d_samp, out)
    info = {"n_rois": int(B), "chains": chains, "draws_per_chain": int(S), "n_burn": n_burn,
            "n_eff": float(LP.n_eff(b)), "kappa_median": float(kappa.median()),
            "acceptance_median": float(acc.median()), "rhat_max": float(rhat.max()), "rhat_median": float(rhat.median()),
            "frac_rhat_gt_1_1": float((rhat.max(dim=1).values > 1.1).double().mean()),
            "ess_min": float(ess.min()), "seconds": round(time.time() - t0, 1)}
    (out / "uq_info.json").write_text(json.dumps(info, indent=2), encoding="utf8")
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int)
    p.add_argument("--burn", type=int, default=3000)
    p.add_argument("--keep", type=int, default=6000)
    p.add_argument("--chains", type=int, default=4)
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("out_root", a.out_root),) if v})
    root = cfg.resolve("out_root") / cfg.name
    pids = a.patients or list_patients(root)
    jobs = [(root / subject_id(pid), a.burn, a.keep, a.chains, int(pid)) for pid in pids
            if (root / subject_id(pid) / "fit_curves.npz").exists()]
    workers = a.workers or os.cpu_count() or 1
    print(f"[{cfg.name}] UQ on {len(jobs)} subjects, {workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, info in ex.map(run_subject, jobs):
            print(f"  {name}: {info['n_rois']} ROIs, acc {info['acceptance_median']:.2f}, R-hat max {info['rhat_max']:.3f} "
                  f"({100 * info['frac_rhat_gt_1_1']:.0f}% ROIs > 1.1), {info['seconds']}s", flush=True)
    print(f"[{cfg.name}] UQ done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
