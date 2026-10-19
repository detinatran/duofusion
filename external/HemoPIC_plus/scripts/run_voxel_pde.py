"""Transport model at the voxel level (run after train_npde.py).

python scripts/run_voxel_pde.py configs/full_aif.yaml --workers 10 [--prior-scale 1] [--no-save]

For every subject and every ROI, the regional fit fixes R, tau, F, the input
dispersion, Pe and the closure split, and each voxel curve gets its own V (shape,
MTT), Cin (amplitude, CBV) and delay with priors centred on the ROI
(npde/voxel.py). Two variants:
  tank  HemoPIC+'s compartment (ROI fits of <name>),
  npde  the transport PDE with the neural closure of the held-out fold
        (<name>__npde/npde_theta.npz).
Maps: CBF, CBV, MTT (anchored on the ROI, see npde/voxel.py), the voxel delay
and the voxel nMSE, written to <out_root>/<name>__vox_<variant>[_p<scale>]/training_XX/.
Every patient is also scored at once (lesion ROC AUC relative to the brain median,
as compare_osvd.py) against oSVD with the same AIF, the ROI-level maps, the voxel
time to peak and the oSVD Tmax. Writes <out_root>/vox_pde[_p<scale>]/{per_patient.csv, summary.json}.
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
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from hemopic_plus.aif import select_aif  # noqa: E402
from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, load_nifti, save_nifti_like, squeeze_ctc, subject_id  # noqa: E402
from hemopic_plus.npde.uqbatch import load_npde_subject  # noqa: E402
from hemopic_plus.npde.voxel import RoiModel, fit_voxels, smooth_curves, voxel_maps  # noqa: E402
from hemopic_plus.osvd import osvd_maps  # noqa: E402
from hemopic_plus.signal import positive_excursion  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402

VARIANTS = ("tank", "npde")
MAPS = (("CBF", "low"), ("CBV", "low"), ("MTT", "high"))


def auc_map(vol, brain, ot, direction):
    if direction == "raw":
        ok = brain & np.isfinite(vol)
        return roc_auc(ot[ok].astype(np.uint8), vol[ok].astype(np.float64)) if ok.any() else np.nan
    valid, score = _lesion_scores(vol, brain, direction)
    return roc_auc(ot[valid], score) if score is not None else np.nan


def work(args):
    pid, a = args
    torch.set_num_threads(1)
    t0 = time.time()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out, name, sub = cfg.resolve("out_root"), cfg.name, subject_id(pid)
    fit_dir, npde_dir = out / name / sub, out / f"{name}__npde" / sub
    if not (npde_dir / "npde_theta.npz").exists():
        return {"patient": pid, "skipped": "no neural PDE fit"}
    base = cfg.resolve("data_root") / sub
    _, raw = load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))
    ctc = squeeze_ctc(raw)
    seg_img, seg = load_nifti(base / SEG4_REL)
    seg = seg.astype(int)
    ot = load_nifti(find_nii(base, "OT"))[1] > 0
    lab = nib.load(str(fit_dir / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
    brain = (seg > 0) & np.all(np.isfinite(ctc), axis=3)
    bt = load_subject(fit_dir)
    bn = load_npde_subject(fit_dir, npde_dir)
    rcfg = bt.cfg
    t, dt, shape, w = bt.t.numpy(), bt.dt, bt.shape.numpy(), bt.w[0].numpy()
    assert ctc.shape[3] == t.size
    s = float(a.prior_scale)
    res = {"patient": pid}
    vols = {}
    for var in VARIANTS:
        v = {k: np.full(seg.shape, np.nan) for k in ("CBF", "CBV", "MTT", "delay", "nmse")}
        for i, rid in enumerate(bt.roi_ids):
            m = brain & (lab == int(rid))
            if m.sum() == 0:
                continue
            if var == "tank":
                th = np.r_[bt.theta_fit[i].numpy(), 0.0]
                model = RoiModel(rcfg, t, dt, shape, th, n_cells=1)
            else:
                model = RoiModel(rcfg, t, dt, shape, bn.theta_fit[i].numpy(), n_cells=bn.n_cells,
                                 nu=bn.nu[i], phi=bn.phi[i])
            c = smooth_curves(ctc[m], rcfg.smooth_sigma_frames, rcfg.baseline_frames, rcfg.clip_nonneg)
            x, _, data = fit_voxels(model, c, w, lam_v=0.02 * s, lam_c=0.02 * s, lam_d=0.002 * s, iters=a.iters)
            cbf, cbv, mtt = voxel_maps(model, x, rcfg.tail_frac)
            for k, val in (("CBF", cbf), ("CBV", cbv), ("MTT", mtt), ("delay", x[:, 2]), ("nmse", data)):
                v[k][m] = val
        vols[var] = v
        for mname, direction in MAPS:
            res[f"vox_{var} {mname}"] = auc_map(v[mname], seg > 0, ot, direction)
        res[f"vox_{var} delay"] = auc_map(v["delay"], seg > 0, ot, "raw")
        res[f"vox_{var} nMSE brain"] = float(np.nanmedian(v["nmse"][brain]))
        res[f"vox_{var} nMSE lesion"] = float(np.nanmedian(v["nmse"][brain & ot])) if (brain & ot).any() else np.nan
        if a.save:
            d = out / f"{name}__vox_{var}{a.tag}" / sub
            d.mkdir(parents=True, exist_ok=True)
            for k, vol in v.items():
                save_nifti_like(seg_img, vol, d / f"VOX_{k}.nii.gz")
    for src, pat in (("ROI tank", out / name / sub / "HemoPIC_{}.nii.gz"), ("ROI npde", npde_dir / "HemoPIC_{}.nii.gz"),
                     ("oSVD", out / "osvd" / sub / "oSVD_{}.nii.gz")):
        for mname, direction in MAPS:
            f = Path(str(pat).format(mname))
            res[f"{src} {mname}"] = auc_map(nib.load(str(f)).get_fdata(), seg > 0, ot, direction) if f.exists() else np.nan
    curves = positive_excursion(ctc[brain].astype(np.float64), cfg.baseline_frames)
    ttp = np.full(seg.shape, np.nan)
    ttp[brain] = np.argmax(gaussian_filter1d(curves, 1.0, axis=1), axis=1) * dt
    res["TTP"] = auc_map(ttp, brain, ot, "raw")
    aif = select_aif(ctc, brain, t, cfg, seed=int(cfg.seed) + pid)
    res["oSVD Tmax"] = np.nan
    if aif is not None:
        art = positive_excursion(aif["aif"][None, :], cfg.baseline_frames)[0]
        *_, tmax = osvd_maps(curves, art, dt, k_h=cfg.k_h, rho=cfg.rho_brain, return_tmax=True)
        vol = np.full(seg.shape, np.nan)
        vol[brain] = tmax
        res["oSVD Tmax"] = auc_map(vol, brain, ot, "raw")
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--prior-scale", type=float, default=1.0, help="multiplies the shrinkage weights (0.02, 0.02, 0.002)")
    p.add_argument("--iters", type=int, default=30, help="Levenberg-Marquardt iterations")
    p.add_argument("--no-save", dest="save", action="store_false")
    a = p.parse_args()
    a.tag = "" if a.prior_scale == 1.0 else f"_p{a.prior_scale:g}"
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out = cfg.resolve("out_root")
    od = out / f"vox_pde{a.tag}"
    od.mkdir(parents=True, exist_ok=True)
    pids = sorted(a.patients or list_patients(cfg.resolve("data_root")))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, [(pid, a) for pid in pids]):
            if "skipped" in r:
                print(f"  training_{r['patient']}: skipped ({r['skipped']})", flush=True)
                continue
            rows.append(r)
            print(f"  training_{r['patient']:<3d} {r['seconds']:6.0f}s  npde CBF {r['vox_npde CBF']:.3f} MTT {r['vox_npde MTT']:.3f} "
                  f"delay {r['vox_npde delay']:.3f} | oSVD CBF {r['oSVD CBF']:.3f} MTT {r['oSVD MTT']:.3f} | TTP {r['TTP']:.3f}", flush=True)
    keys = [k for k in rows[0] if k not in ("patient", "seconds")]
    with open(od / "per_patient.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.writer(f)
        wr.writerow(["patient", *keys])
        for r in rows:
            wr.writerow([r["patient"], *[r.get(k) for k in keys]])
    common = [r for r in rows if np.isfinite(r.get("oSVD CBF", np.nan))]

    def mean(k, rs):
        return float(np.nanmean([r[k] for r in rs]))

    def paired(x, y, rs):
        d = np.array([r[x] - r[y] for r in rs])
        d = d[np.isfinite(d)]
        return {"diff": float(d.mean()), "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, "n": int(d.size)}

    summ = {"n_all": len(rows), "n_osvd": len(common), "prior_scale": a.prior_scale,
            "mean_all": {k: mean(k, rows) for k in keys}, "mean_osvd_patients": {k: mean(k, common) for k in keys}}
    tests = {}
    for var in VARIANTS:
        for m in ("CBF", "CBV", "MTT"):
            tests[f"vox_{var} - oSVD {m}"] = paired(f"vox_{var} {m}", f"oSVD {m}", common)
            tests[f"vox_{var} - ROI {var} {m}"] = paired(f"vox_{var} {m}", f"ROI {var} {m}", rows)
        tests[f"vox_{var} delay - oSVD Tmax"] = paired(f"vox_{var} delay", "oSVD Tmax", common)
        tests[f"vox_{var} delay - TTP"] = paired(f"vox_{var} delay", "TTP", rows)
    for m in ("CBF", "CBV", "MTT", "delay"):
        tests[f"vox_npde - vox_tank {m}"] = paired(f"vox_npde {m}", f"vox_tank {m}", rows)
    tests["vox_npde - vox_tank nMSE brain"] = paired("vox_npde nMSE brain", "vox_tank nMSE brain", rows)
    summ["paired"] = tests
    (od / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf8")
    print(f"lesion ROC AUC, mean over {len(common)} patients with oSVD maps (all {len(rows)} in brackets)")
    for k in keys:
        print(f"  {k:28s} {summ['mean_osvd_patients'][k]:.4f}  ({summ['mean_all'][k]:.4f})")
    for k, v in tests.items():
        print(f"  {k:36s} {v['diff']:+.4f}  p={v['p']:.2g}  n={v['n']}")


if __name__ == "__main__":
    main()
