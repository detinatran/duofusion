"""Uncertainty of the voxel-level fit (twin-vox), run after run_uq.py.

python scripts/run_voxel_uq.py configs/full_aif.yaml [--voxels 3000] [--check 600] [--workers 3]

For a stratified sample of voxels of every patient (a third in the final
infarct when possible), with the compartment variant of run_voxel_pde.py:
  1. voxel posterior: tempered Laplace of (log V, log Cin, delay) with
     kappa_v = n_eff D_v / (2 N_v) from the voxel's own residual (as for the ROIs,
     dof 0) and with the residual degrees of freedom of the voxel fit (n_eff - 3);
  2. ROI uncertainty passed down: dx/dtheta by the implicit function theorem and the
     covariance of the ROI posterior draws (uq/posterior.npz); log MTT_v =
     log 60 + log V_v - log CBF_roi(theta), so both parts add on the log scale;
  3. how much the data decides: posterior over prior standard deviation of log V;
  4. checks on a subset: semi-synthetic coverage of a known truth (noise with the
     level and temporal correlation of the voxel residuals, refit) and predictive
     coverage of hidden blocks of time frames (5 folds), as eval_semisynth.py and
     eval_heldout.py do for the ROIs;
  5. for --lin-patients: the linearised ROI part against refits under --draws ROI
     posterior draws.
--full-maps writes, for the listed patients and all brain voxels, the maps
VOX_MTT_sd_log, VOX_MTT_sd_roi_log, VOX_MTT_ci95_rel and VOX_V_post_prior_ratio to
<out_root>/<name>__vox_tank/training_XX/uq/.
Writes <out_root>/vox_uq<tag>/{per_patient.csv, summary.json, pooled.npz}.
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
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, load_nifti, save_nifti_like, squeeze_ctc, subject_id  # noqa: E402
from hemopic_plus.npde.voxel import RoiModel, fit_voxels, roi_cbf, roi_sensitivity, smooth_curves, voxel_laplace  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE  # noqa: E402

Z95 = 1.959964
LEVELS = (0.5, 0.8, 0.95)
NAMES7 = ["log_R", "log_tau", "log_V", "log_F", "log_Cin", "delay", "log_disp"]


def rel_width(sd_log):
    return np.exp(Z95 * sd_log) - np.exp(-Z95 * sd_log)


def sample_voxels(rng, lab, brain, ot, n):
    ok = brain & (lab > 0)
    les = np.argwhere(ok & ot)
    rest = np.argwhere(ok & ~ot)
    n_les = min(len(les), n // 3)
    pick = [les[rng.choice(len(les), n_les, replace=False)]] if n_les else []
    pick.append(rest[rng.choice(len(rest), min(len(rest), n - n_les), replace=False)])
    return np.concatenate(pick)


def roi_covariance(post, rid):
    names = [str(s) for s in post["names"]]
    assert names == NAMES7, names
    th = post["theta"][list(post["roi_ids"]).index(rid)]
    cov = np.zeros((8, 8))
    cov[:7, :7] = np.cov(th, rowvar=False)
    return cov, th


def draws_cov(x, cov, n, rng):
    vals, vecs = np.linalg.eigh(cov)
    L = vecs * np.sqrt(np.clip(vals, 0.0, None))[:, None, :]
    z = rng.standard_normal((n, x.shape[0], 3))
    return x[None, :, :] + np.einsum("nij,snj->sni", L, z)


def checks(model, x, c, w, cfg, rng):
    """Semi-synthetic coverage and hidden-frame coverage on the voxels (x, c) of one ROI."""
    out = {}
    with torch.no_grad():
        C_true = model.simulate(torch.as_tensor(x, dtype=DTYPE))[1].numpy()
    num = (w[None, :] * (C_true - c) ** 2).mean(axis=1)
    e = gaussian_filter1d(rng.standard_normal(C_true.shape), sigma=float(cfg.smooth_sigma_frames), axis=1, mode="nearest")
    e *= np.sqrt(num / np.maximum((w[None, :] * e ** 2).mean(axis=1), 1e-30))[:, None]
    xh, _, _ = fit_voxels(model, C_true + e, w)
    for dof in (0, 3):
        ch, _, _ = voxel_laplace(model, xh, C_true + e, w, sigma_s=cfg.smooth_sigma_frames, dof=dof)
        for k, name in enumerate(("logV", "logCin", "delay")):
            out[f"semisynth {name} dof{dof}"] = np.abs(xh[:, k] - x[:, k]) <= Z95 * np.sqrt(np.maximum(ch[:, k, k], 0.0))
    T = c.shape[1]
    edges = np.linspace(0, T, 6).astype(int)
    hits = {(dof, lev): [] for dof in (0, 3) for lev in LEVELS}
    for f in range(5):
        hide = np.zeros(T, bool)
        hide[edges[f]:edges[f + 1]] = True
        wk = np.where(hide, 0.0, w)
        xk, _, _ = fit_voxels(model, c, wk)
        with torch.no_grad():
            Ck = model.simulate(torch.as_tensor(xk, dtype=DTYPE))[1].numpy()
        n_train = int((~hide).sum())
        n_eff = n_train / (2.0 * np.sqrt(np.pi) * float(cfg.smooth_sigma_frames))
        rss = (wk[None, :] * (Ck - c) ** 2).sum(axis=1) / n_train
        for dof in (0, 3):
            ck, _, _ = voxel_laplace(model, xk, c, wk, sigma_s=cfg.smooth_sigma_frames, dof=dof)
            xs = draws_cov(xk, ck, 100, rng)
            with torch.no_grad():
                Cs = model.simulate(torch.as_tensor(xs.reshape(-1, 3), dtype=DTYPE))[1].numpy().reshape(100, xk.shape[0], T)
            sigma2 = rss * (n_eff / max(n_eff - dof, 0.5))
            noise = np.sqrt(sigma2[:, None] / np.maximum(w[None, hide], 1e-6))
            pred = Cs[:, :, hide] + noise[None] * rng.standard_normal(Cs[:, :, hide].shape)
            obs = c[:, hide]
            for lev in LEVELS:
                lo, hi = np.quantile(pred, [(1 - lev) / 2, 1 - (1 - lev) / 2], axis=0)
                hits[(dof, lev)].append(((obs >= lo) & (obs <= hi)).mean(axis=1))
    for (dof, lev), v in hits.items():
        out[f"heldout {lev:.2f} dof{dof}"] = np.mean(v, axis=0)
    return out


def work(args):
    pid, a = args
    torch.set_num_threads(1)
    t0 = time.time()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out, name, sub = cfg.resolve("out_root"), cfg.name, subject_id(pid)
    fit_dir = out / name / sub
    post_f = fit_dir / "uq" / "posterior.npz"
    if not post_f.exists():
        return {"patient": pid, "skipped": "no ROI posterior"}
    base = cfg.resolve("data_root") / sub
    ctc = squeeze_ctc(load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))[1])
    seg_img, seg = load_nifti(base / SEG4_REL)
    seg = seg.astype(int)
    ot = load_nifti(find_nii(base, "OT"))[1] > 0
    lab = load_nifti(fit_dir / "roi_labels_seg4_within_kmeans.nii.gz")[1].astype(int)
    brain = (seg > 0) & np.all(np.isfinite(ctc), axis=3)
    b = load_subject(fit_dir)
    rcfg = b.cfg
    t, dt, shape, w = b.t.numpy(), b.dt, b.shape.numpy(), b.w[0].numpy()
    post = np.load(post_f)
    rng = np.random.default_rng(pid)
    full = pid in (a.full_maps or [])
    pick = np.argwhere(brain & (lab > 0)) if full else sample_voxels(rng, lab, brain, ot, a.voxels)
    region = np.where(ot[tuple(pick.T)], 2, np.where(seg[tuple(pick.T)] == 1, 0, np.where(seg[tuple(pick.T)] == 2, 1, 3)))
    vlab = lab[tuple(pick.T)]
    rec = {k: np.full(len(pick), np.nan) for k in ("sd_vox0", "sd_vox3", "sd_roi", "ratio0", "ratio3", "mtt")}
    chk = {}
    lin = {"emp": [], "lin": []}
    n_check_left = a.check
    for i, rid in enumerate(b.roi_ids):
        sel = np.where(vlab == int(rid))[0]
        if sel.size == 0:
            continue
        th = np.r_[b.theta_fit[i].numpy(), 0.0]

        def make(th_):
            return RoiModel(rcfg, t, dt, shape, th_, n_cells=1)
        model = make(th)
        c = smooth_curves(ctc[tuple(pick[sel].T)], rcfg.smooth_sigma_frames, rcfg.baseline_frames, rcfg.clip_nonneg)
        x, _, _ = fit_voxels(model, c, w)
        cbf_roi = roi_cbf(model, rcfg.tail_frac)
        rec["mtt"][sel] = 60.0 * np.exp(x[:, 0]) / cbf_roi
        for dof in (0, 3):
            cov, _, psd = voxel_laplace(model, x, c, w, sigma_s=rcfg.smooth_sigma_frames, dof=dof)
            sd = np.sqrt(np.maximum(cov[:, 0, 0], 0.0))
            rec[f"sd_vox{dof}"][sel] = sd
            rec[f"ratio{dof}"][sel] = sd / psd[:, 0]
        dx, dl = roi_sensitivity(make, th, x, c, w, rcfg.tail_frac)
        S, th_draws = roi_covariance(post, int(rid))
        g = dx[:, 0, :] - dl[None, :]
        rec["sd_roi"][sel] = np.sqrt(np.maximum(np.einsum("np,pq,nq->n", g, S, g), 0.0))
        if n_check_left > 0 and not full:
            m = min(sel.size, max(20, int(round(a.check * sel.size / len(pick)))), n_check_left)
            r = checks(model, x[:m], c[:m], w, rcfg, rng)
            for k, v in r.items():
                chk.setdefault(k, []).append(v)
            chk.setdefault("region", []).append(region[sel[:m]])
            n_check_left -= m
        if pid in (a.lin_patients or []) and not full:
            m = min(sel.size, 200)
            idx = rng.choice(th_draws.shape[0], a.draws, replace=False)
            lm = []
            for s in idx:
                th_s = np.r_[th_draws[s], 0.0]
                ms = make(th_s)
                xs, _, _ = fit_voxels(ms, c[:m], w)
                lm.append(xs[:, 0] - np.log(roi_cbf(ms, rcfg.tail_frac)))
            lin["emp"].append(np.std(np.array(lm), axis=0, ddof=1))
            lin["lin"].append(rec["sd_roi"][sel[:m]])
    res = {"patient": pid, "seconds": round(time.time() - t0, 1), "region": region, **rec,
           "checks": {k: np.concatenate(v) for k, v in chk.items()}}
    if lin["emp"]:
        res["lin_emp"] = np.concatenate(lin["emp"])
        res["lin_lin"] = np.concatenate(lin["lin"])
    if full:
        d = out / f"{name}__vox_tank" / sub / "uq"
        d.mkdir(parents=True, exist_ok=True)
        sd_tot = np.sqrt(rec["sd_vox3"] ** 2 + rec["sd_roi"] ** 2)
        for key, val in (("VOX_MTT_sd_log", sd_tot), ("VOX_MTT_sd_roi_log", rec["sd_roi"]),
                         ("VOX_MTT_ci95_rel", rel_width(sd_tot)), ("VOX_V_post_prior_ratio", rec["ratio3"])):
            vol = np.full(seg.shape, np.nan)
            vol[tuple(pick.T)] = val
            save_nifti_like(seg_img, vol, d / f"{key}.nii.gz")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--voxels", type=int, default=3000)
    p.add_argument("--check", type=int, default=600)
    p.add_argument("--lin-patients", type=int, nargs="*", default=[4, 10, 22])
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--full-maps", type=int, nargs="*", help="patients for which every brain voxel is processed and maps are written")
    p.add_argument("--tag", default="", help="suffix of the results folder <out_root>/vox_uq<tag>")
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out = cfg.resolve("out_root")
    pids = sorted(a.patients or list_patients(cfg.resolve("data_root")))
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, [(pid, a) for pid in pids]):
            if "skipped" in r:
                print(f"  training_{r['patient']}: skipped ({r['skipped']})", flush=True)
                continue
            results.append(r)
            ok = r["region"] < 3
            print(f"  training_{r['patient']:<3d} {r['seconds']:6.0f}s  MTT 95% width (total, dof3) "
                  f"{np.nanmedian(rel_width(np.sqrt(r['sd_vox3'][ok] ** 2 + r['sd_roi'][ok] ** 2))):.2f}  "
                  f"post/prior {np.nanmedian(r['ratio3'][ok]):.2f}", flush=True)
    if not results:
        sys.exit("no patient with an ROI posterior")
    if a.full_maps and set(pids) <= set(a.full_maps):
        return
    od = out / f"vox_uq{a.tag}"
    od.mkdir(parents=True, exist_ok=True)
    regions = {"GM": 0, "WM": 1, "lesion": 2}
    pooled = {k: np.concatenate([r[k] for r in results]) for k in ("region", "sd_vox0", "sd_vox3", "sd_roi", "ratio0", "ratio3", "mtt")}
    pooled["patient"] = np.concatenate([np.full(len(r["region"]), r["patient"]) for r in results])
    np.savez_compressed(od / "pooled.npz", **pooled)
    summ = {"n_patients": len(results), "n_voxels": int(len(pooled["region"]))}
    for key, code in [("all", None), *regions.items()]:
        m = (pooled["region"] < 3) if code is None else (pooled["region"] == code)
        s0, s3, sr = pooled["sd_vox0"][m], pooled["sd_vox3"][m], pooled["sd_roi"][m]
        summ[key] = {"n": int(m.sum()),
                     "MTT ci95 rel width, voxel part, dof0": float(np.nanmedian(rel_width(s0))),
                     "MTT ci95 rel width, voxel part, dof3": float(np.nanmedian(rel_width(s3))),
                     "MTT ci95 rel width, ROI part": float(np.nanmedian(rel_width(sr))),
                     "MTT ci95 rel width, total, dof3": float(np.nanmedian(rel_width(np.sqrt(s3 ** 2 + sr ** 2)))),
                     "share of variance from the ROI, dof3": float(np.nanmedian(sr ** 2 / (s3 ** 2 + sr ** 2))),
                     "posterior/prior sd of log V, median, dof3": float(np.nanmedian(pooled["ratio3"][m])),
                     "fraction data-determined (ratio < 0.5), dof3": float(np.nanmean(pooled["ratio3"][m] < 0.5)),
                     "fraction prior-dominated (ratio > 0.9), dof3": float(np.nanmean(pooled["ratio3"][m] > 0.9))}
    chk_keys = sorted({k for r in results for k in r["checks"] if k != "region"})
    creg = np.concatenate([r["checks"]["region"] for r in results if "region" in r["checks"]] or [np.zeros(0, int)])
    summ["checks"] = {}
    for k in chk_keys:
        v = np.concatenate([r["checks"][k] for r in results if k in r["checks"]]).astype(float)
        summ["checks"][k] = {"all": float(np.mean(v)), **{rn: float(np.mean(v[creg == c])) for rn, c in regions.items() if np.any(creg == c)}}
    lin = [r for r in results if "lin_emp" in r]
    if lin:
        emp = np.concatenate([r["lin_emp"] for r in lin])
        lnz = np.concatenate([r["lin_lin"] for r in lin])
        ok = np.isfinite(emp) & np.isfinite(lnz) & (lnz > 0)
        summ["linearisation"] = {"patients": [r["patient"] for r in lin], "n_voxels": int(ok.sum()),
                                 "median refit sd / linearised sd": float(np.median(emp[ok] / lnz[ok])),
                                 "mean refit var / linearised var": float(np.mean(emp[ok] ** 2 / lnz[ok] ** 2)),
                                 "median refit sd": float(np.median(emp[ok])), "median linearised sd": float(np.median(lnz[ok])),
                                 "spearman": float(np.corrcoef(np.argsort(np.argsort(emp[ok])), np.argsort(np.argsort(lnz[ok])))[0, 1])}
    (od / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf8")
    with open(od / "per_patient.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.writer(f)
        wr.writerow(["patient", "n", "MTT_ci95_rel_total_dof3", "MTT_ci95_rel_voxel_dof3", "MTT_ci95_rel_roi", "post_prior_ratio_dof3"])
        for r in results:
            ok = r["region"] < 3
            wr.writerow([r["patient"], int(ok.sum()),
                         float(np.nanmedian(rel_width(np.sqrt(r["sd_vox3"][ok] ** 2 + r["sd_roi"][ok] ** 2)))),
                         float(np.nanmedian(rel_width(r["sd_vox3"][ok]))), float(np.nanmedian(rel_width(r["sd_roi"][ok]))),
                         float(np.nanmedian(r["ratio3"][ok]))])
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
