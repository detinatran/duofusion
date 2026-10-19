"""Deconvolution core and hypoperfusion volumes with the AIF of the HemoPIC+ fit, the
comparators of the truncation study.

python scripts/osvd_volumes.py <data_root> <twin_out> <out_dir> [--method osvd] [--sigma 3] [--workers 8]

--method osvd        block-circulant SVD with oscillation-index threshold (HemoPIC_plus)
         ssvd        standard truncated SVD, threshold 20% (duofusion/deconv.py)
         gamma_osvd  tissue curves and AIF extended to --extend-s seconds by a gamma variate
                     fitted to the first pass, then oSVD (the classic fix for truncation)
         gamma_ssvd  the same extension, then sSVD
         learned_ssvd  curves and AIF extended to 70 s by the learned network
                     (duofusion/learned.py; --models, --folds: the model of the patient's
                     held-out fold, otherwise the mean of the five), then sSVD

oSVD CBF and Tmax (external/HemoPIC_plus/hemopic_plus/osvd.py) on the tissue voxels,
smoothed as in duofusion_volumes.py. This oSVD Tmax is offset from the vendor Tmax (it is
timed from the start of the HemoPIC+ AIF), so, exactly as for the twin, Tmax is taken relative
to the median of the less delayed hemisphere: healthy = Tmax_rel < 2 s, hypo = Tmax_rel > d
on DT_GRID; core = rCBF < tau with the contralateral healthy median.
Writes <out_dir>/training_X.npz with core_grid (TAU_GRID), hypo_grid (DT_GRID) and
union_grid, like duofusion_volumes.py.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "HemoPIC_plus"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from duofusion.data import load_subject  # noqa: E402
from duofusion.deconv import gamma_extend, ssvd_maps  # noqa: E402
from duofusion import learned  # noqa: E402
from duofusion.reference import affected_side, nan_gaussian, spacing_of  # noqa: E402
from hemopic_plus.osvd import osvd_maps  # noqa: E402
from hemopic_plus.signal import positive_excursion  # noqa: E402
from hemopic_plus.uq.data import load_subject as load_fit  # noqa: E402

TAU_GRID = np.round(np.arange(0.10, 0.551, 0.05), 3)
DT_GRID = np.round(np.arange(0.0, 8.01, 0.5), 2)


def gpu_free(busy, samples=3):
    """True when no sample of the GPU utilization (nvidia-smi) reaches busy %."""
    import subprocess
    import time
    for _ in range(samples):
        try:
            u = float(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.split()[0])
        except Exception:  # noqa: BLE001
            return False
        if u >= busy:
            return False
        time.sleep(0.5)
    return True


def work(args):
    data_root, duo, out, num, sigma, method, extend_s, models, fold_map, device = args
    fit_dir = duo / "isles24_full_aif" / f"training_{num}"
    if not (fit_dir / "fit_curves.npz").exists():
        return {"patient": num, "skipped": "no fit"}
    s = load_subject(data_root, num)
    cfg = load_fit(fit_dir).cfg
    info = json.loads((fit_dir / "run_info_core.json").read_text(encoding="utf8"))
    ctc = s.ctc()
    lab = np.rint(nib.load(str(fit_dir / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata()).astype(int)
    brain = (s.tissue > 0) & np.all(np.isfinite(ctc), axis=3) & (lab > 0)
    art_raw = np.load(fit_dir / "fit_curves.npz")["cart_norm"]
    dt = float(info["dt_seconds"])
    if method == "learned_ssvd":
        ms = learned.load_models(models, [fold_map[num]] if num in fold_map else None, device)
        raw = learned.extend(ctc[brain], ms, device)
        art_raw = learned.extend(art_raw[None, :], ms, device)[0]
        curves = positive_excursion(raw.astype(np.float64), cfg.baseline_frames)
    else:
        curves = positive_excursion(ctc[brain].astype(np.float64), cfg.baseline_frames)
    art = positive_excursion(art_raw[None, :], cfg.baseline_frames)[0]
    if method in ("gamma_osvd", "gamma_ssvd"):
        curves = gamma_extend(curves, dt, extend_s)
        art = gamma_extend(art[None, :], dt, extend_s)[0]
    if method in ("ssvd", "gamma_ssvd", "learned_ssvd"):
        cbf_v, tmax_v = ssvd_maps(curves, art, dt, k_h=cfg.k_h, rho=cfg.rho_brain)
    else:
        cbf_v, _, _, tmax_v = osvd_maps(curves, art, dt, k_h=cfg.k_h, rho=cfg.rho_brain, return_tmax=True)
    cbf, tmax = np.full(brain.shape, np.nan), np.full(brain.shape, np.nan)
    cbf[brain], tmax[brain] = cbf_v, tmax_v
    sp = spacing_of(s.affine)
    ok = brain & np.isfinite(cbf) & (cbf > 0) & np.isfinite(tmax)
    if sigma > 0:
        cbf, tmax = nan_gaussian(np.where(ok, cbf, np.nan), sigma, sp), nan_gaussian(np.where(ok, tmax, np.nan), sigma, sp)
    valid = brain & np.isfinite(cbf) & (cbf > 0) & np.isfinite(tmax)
    med = [np.median(tmax[valid & (s.hemi == h)]) if np.any(valid & (s.hemi == h)) else np.inf for h in (1, 2)]
    trel = tmax - (min(med) if np.isfinite(min(med)) else np.median(tmax[valid]))
    healthy = valid & (trel < 2.0)
    side = affected_side(s, valid & (trel > 4.0))
    contra = (s.hemi == 3 - side) if side else (s.hemi > 0)
    sel = healthy & contra if np.any(healthy & contra) else (healthy if np.any(healthy) else valid)
    r = cbf / float(np.median(cbf[sel]))
    cores = [valid & (r < t) for t in TAU_GRID]
    hypos = [valid & (trel > t) for t in DT_GRID]
    ml = s.voxel_ml
    np.savez_compressed(out / f"training_{num}.npz", tau_grid=TAU_GRID, dt_grid=DT_GRID,
                        core_grid=np.array([c.sum() for c in cores]) * ml,
                        hypo_grid=np.array([h.sum() for h in hypos]) * ml,
                        union_grid=np.array([[(c | h).sum() for h in hypos] for c in cores]) * ml)
    out_dir = out / f"training_{num}"
    out_dir.mkdir(exist_ok=True)
    tag = method.upper()
    for name, vol in ((f"{tag}_CBF", cbf), (f"{tag}_Tmax_rel", trel)):
        nib.save(nib.Nifti1Image(np.where(valid, vol, np.nan).astype(np.float32), s.affine), str(out_dir / f"{name}.nii.gz"))
    return {"patient": num, "device": device, "core30_ml": float(cores[4].sum()) * ml, "hypo_rel2_ml": float(hypos[4].sum()) * ml}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=Path)
    p.add_argument("twin_out", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--method", choices=("osvd", "ssvd", "gamma_osvd", "gamma_ssvd", "learned_ssvd"), default="osvd")
    p.add_argument("--models", type=Path, default=Path.home() / "duofusion_data" / "learned_extrap")
    p.add_argument("--folds", type=Path, default=None, help="JSON patient -> held-out fold (ISLES'24)")
    p.add_argument("--device", default="cpu", help="cpu, cuda, or auto: cuda only while the GPU is not busy (< --busy %)")
    p.add_argument("--busy", type=float, default=90.0, help="auto: GPU utilization (%) at or above which the CPU is used")
    p.add_argument("--threads", type=int, default=12, help="torch threads for learned_ssvd on the CPU")
    p.add_argument("--extend-s", type=float, default=70.0, help="gamma_osvd: length of the extended curves (s)")
    a = p.parse_args()
    nums = sorted(int(d.name.split("_")[1]) for d in (a.twin_out / "isles24_full_aif").glob("training_*"))
    fold_map = {int(k): v for k, v in json.loads(a.folds.read_text()).items()} if a.folds else {}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(a.data_root, a.twin_out, a.out_dir, n, a.sigma, a.method, a.extend_s, a.models, fold_map, a.device)
            for n in nums if not (a.out_dir / f"training_{n}.npz").exists()]
    if a.method == "learned_ssvd":
        # one process with a multi-threaded network (CUDA in pool workers hangs under WSL)
        __import__("torch").set_num_threads(a.threads)
        if a.device == "auto":
            def pick(job):
                return work((*job[:-1], "cuda" if gpu_free(a.busy) else "cpu"))
            results = map(pick, jobs)
        else:
            results = map(work, jobs)
    else:
        results = ProcessPoolExecutor(max_workers=a.workers).map(work, jobs)
    for r in results:
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)


if __name__ == "__main__":
    main()
