"""External test set (ISLES'24 CT perfusion, scripts/prepare_isles24.py): lesion AUC of
every method, with the settings fixed on ISLES 2017.

python scripts/eval_external.py --data-root <ISLES2024_Training> [--prefix isles24] [--workers 24] [--voxel]

Needs run_cohort.py fits of <prefix>_baseline (HemoPIC), <prefix>_full_aif (HemoPIC+,
compartment) and <prefix>_full_aif_idvox_osvd_nc (HemoPIC+ with indicator-dilution
CBV and the oSVD CBF texture). Per patient:
  * twin-vox: the voxel-level compartment fit of npde/voxel.py with the ROI fits
    of <prefix>_full_aif and the priors of ISLES 2017, maps written to
    <out_root>/<prefix>_full_aif__vox_tank/ (kept when present unless --voxel);
    CBF = 60 CBV / MTT with the indicator-dilution CBV, as on ISLES 2017;
  * oSVD with the AIF of the HemoPIC+ fit;
  * the vendor maps of the challenge (MR_rCBF, MR_rCBV, MR_MTT).
Lesion ROC AUC of CBF (low), CBV (low) and MTT (high) relative to the brain
median, as compare_osvd.py. Writes <out_root>/<prefix>_external/{per_patient.csv, summary.json}.
--metrics also scores average precision, Dice and lesion volume (hemopic_plus/lesion_metrics.py)
and writes <prefix>_external/{lesion_metrics.csv, lesion_metrics.json}.
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
from scipy.stats import wilcoxon  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.lesion_metrics import map_metrics, paired_bootstrap, summarize  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, load_nifti, save_nifti_like, squeeze_ctc, subject_id  # noqa: E402
from hemopic_plus.npde.voxel import RoiModel, fit_voxels, smooth_curves, voxel_maps  # noqa: E402
from hemopic_plus.osvd import osvd_maps  # noqa: E402
from hemopic_plus.signal import positive_excursion  # noqa: E402
from hemopic_plus.uq.data import load_subject  # noqa: E402

METHODS = ("vendor", "oSVD", "HemoPIC", "HemoPIC+", "HemoPIC+ ID-CBV oSVD texture", "twin-vox")


def auc(vol, brain, ot, direction):
    valid, score = _lesion_scores(vol, brain, direction)
    return roc_auc(ot[valid], score) if score is not None else np.nan


def voxel_fit(fit_dir, ctc, brain, lab):
    b = load_subject(fit_dir)
    cfg = b.cfg
    t, dt, shape, w = b.t.numpy(), b.dt, b.shape.numpy(), b.w[0].numpy()
    out = {k: np.full(brain.shape, np.nan) for k in ("MTT", "delay", "nmse")}
    for i, rid in enumerate(b.roi_ids):
        m = brain & (lab == int(rid))
        if not m.any():
            continue
        model = RoiModel(cfg, t, dt, shape, np.r_[b.theta_fit[i].numpy(), 0.0], n_cells=1)
        c = smooth_curves(ctc[m], cfg.smooth_sigma_frames, cfg.baseline_frames, cfg.clip_nonneg)
        x, _, data = fit_voxels(model, c, w)
        out["MTT"][m] = voxel_maps(model, x, cfg.tail_frac)[2]
        out["delay"][m], out["nmse"][m] = x[:, 2], data
    return out


def work(args):
    pid, a = args
    torch.set_num_threads(1)
    t0 = time.time()
    cfg = load_config(Path("configs") / f"{a.prefix}_full_aif.yaml", {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out, sub = cfg.resolve("out_root"), subject_id(pid)
    base = cfg.resolve("data_root") / sub
    fit_dir = out / f"{a.prefix}_full_aif" / sub
    if not (fit_dir / "run_info_core.json").exists():
        return {"patient": pid, "skipped": "no HemoPIC+ fit"}
    info = json.loads((fit_dir / "run_info_core.json").read_text(encoding="utf8"))
    seg_img, seg = load_nifti(base / SEG4_REL)
    ot = load_nifti(find_nii(base, "OT"))[1] > 0
    brain = seg > 0
    if not (ot & brain).any():
        return {"patient": pid, "skipped": "no infarct in the CTP field of view"}
    ctc = squeeze_ctc(load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))[1])
    show = brain & np.all(np.isfinite(ctc), axis=3)
    res = {"patient": pid, "arterial_input": info["arterial_input"], "lesion_voxels": int((ot & brain).sum())}
    maps = {"vendor": {m: load_nifti(find_nii(base, f))[1] for m, f in (("CBF", "MR_rCBF"), ("CBV", "MR_rCBV"), ("MTT", "MR_MTT"))}}
    for name, folder in (("HemoPIC", f"{a.prefix}_baseline"), ("HemoPIC+", f"{a.prefix}_full_aif"),
                         ("HemoPIC+ ID-CBV oSVD texture", f"{a.prefix}_full_aif_idvox_osvd_nc")):
        d = out / folder / sub
        if (d / "HemoPIC_MTT.nii.gz").exists():
            maps[name] = {m: nib.load(str(d / f"HemoPIC_{m}.nii.gz")).get_fdata() for m, _ in MAPS}
    art = positive_excursion(np.load(fit_dir / "fit_curves.npz")["cart_norm"][None, :], cfg.baseline_frames)[0]
    curves = positive_excursion(ctc[show].astype(np.float64), cfg.baseline_frames)
    o = dict(zip(("CBF", "CBV", "MTT"), osvd_maps(curves, art, float(info["dt_seconds"]), k_h=cfg.k_h, rho=cfg.rho_brain)))
    maps["oSVD"] = {}
    for m in ("CBF", "CBV", "MTT"):
        v = np.full(seg.shape, np.nan)
        v[show] = o[m]
        maps["oSVD"][m] = v
    vd = out / f"{a.prefix}_full_aif__vox_tank" / sub
    if a.voxel or not (vd / "VOX_MTT.nii.gz").exists():
        lab = nib.load(str(fit_dir / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
        vx = voxel_fit(fit_dir, ctc, show, lab)
        vd.mkdir(parents=True, exist_ok=True)
        for k, v in vx.items():
            save_nifti_like(seg_img, v, vd / f"VOX_{k}.nii.gz")
    if "HemoPIC+ ID-CBV oSVD texture" in maps:
        cbv = maps["HemoPIC+ ID-CBV oSVD texture"]["CBV"]
        mtt = nib.load(str(vd / "VOX_MTT.nii.gz")).get_fdata()
        maps["twin-vox"] = {"CBV": cbv, "MTT": mtt,
                              "CBF": np.where(np.isfinite(mtt) & (mtt > 0), 60.0 * cbv / np.maximum(mtt, 1e-6), np.nan)}
    for name, vols in maps.items():
        for m, direction in MAPS:
            res[f"{name} {m}"] = auc(vols[m], brain, ot, direction)
        res[f"{name} mean"] = float(np.mean([res[f"{name} {m}"] for m, _ in MAPS]))
    if a.metrics:
        voxel_ml = float(np.prod(seg_img.header.get_zooms()[:3])) / 1000.0
        rec = {"patient": pid}
        for name, vols in maps.items():
            rec[name] = {}
            for m, direction in MAPS:
                mm = map_metrics(vols[m], brain, ot, direction, voxel_ml)
                if mm is not None:
                    mm["auc"] = res[f"{name} {m}"]
                    rec[name][m] = mm
        res["_metrics"] = rec
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-root")
    p.add_argument("--prefix", default="isles24")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--voxel", action="store_true", help="refit twin-vox even when its maps exist")
    p.add_argument("--metrics", action="store_true", help="also AP, Dice and lesion volume")
    a = p.parse_args()
    cfg = load_config(Path("configs") / f"{a.prefix}_full_aif.yaml", {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out = cfg.resolve("out_root")
    pids = sorted(a.patients or list_patients(cfg.resolve("data_root")))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, [(pid, a) for pid in pids]):
            if "skipped" in r:
                print(f"  training_{r['patient']}: skipped ({r['skipped']})", flush=True)
                continue
            rows.append(r)
            print(f"  training_{r['patient']:<4d} {r['seconds']:6.0f}s  " + "  ".join(
                f"{k} {r.get(k + ' mean', np.nan):.3f}" for k in METHODS), flush=True)
    od = out / f"{a.prefix}_external"
    od.mkdir(parents=True, exist_ok=True)
    records = [r.pop("_metrics") for r in rows if "_metrics" in r]
    if records:
        mrows, msumm = summarize(records, list(METHODS), MAPS)
        for r in mrows:
            for meth in METHODS:
                r[f"{meth} mean AUC"] = np.nanmean([r.get(f"{meth} {m} AUC", np.nan) for m, _ in MAPS])
                r[f"{meth} mean AP"] = np.nanmean([r.get(f"{meth} {m} AP", np.nan) for m, _ in MAPS])
        msumm["paired, twin-vox minus"] = {
            ref: {k: paired_bootstrap(mrows, f"twin-vox {k}", f"{ref} {k}")
                  for k in ("mean AUC", "mean AP", "CBF Dice CV", "MTT Dice CV")} for ref in METHODS if ref != "twin-vox"}
        mkeys = ["patient"] + sorted({k for r in mrows for k in r if k != "patient"})
        with open(od / "lesion_metrics.csv", "w", newline="", encoding="utf8") as f:
            wr = csv.DictWriter(f, fieldnames=mkeys)
            wr.writeheader()
            wr.writerows(mrows)
        (od / "lesion_metrics.json").write_text(json.dumps(msumm, indent=1, default=float), encoding="utf8")
        for meth, sm in msumm["methods"].items():
            print(f"  {meth:28s} AUC {sm['mean AUC'][0]:.3f} [{sm['mean AUC'][1]:.3f}, {sm['mean AUC'][2]:.3f}]  AP {sm['mean AP'][0]:.3f}  "
                  + "  ".join(f"{m} Dice {sm[m]['Dice (CV threshold)'][0]:.3f}" for m in ("CBF", "MTT") if m in sm), flush=True)
    keys = sorted({k for r in rows for k in r if k not in ("patient",)})
    with open(od / "per_patient.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.writer(f)
        wr.writerow(["patient", *keys])
        for r in rows:
            wr.writerow([r["patient"], *[r.get(k) for k in keys]])
    summ = {}
    for subset, rs in (("all", rows), ("accepted AIF", [r for r in rows if r["arterial_input"] == "auto_aif"])):
        s = {"n": len(rs)}
        for meth in METHODS:
            for m in ("CBF", "CBV", "MTT", "mean"):
                v = [r.get(f"{meth} {m}", np.nan) for r in rs]
                s[f"{meth} {m}"] = float(np.nanmean(v)) if np.isfinite(v).any() else None
        tests = {}
        for x, y in (("HemoPIC+", "HemoPIC"), ("twin-vox", "HemoPIC"), ("HemoPIC+", "oSVD"), ("twin-vox", "oSVD"),
                     ("HemoPIC+ ID-CBV oSVD texture", "oSVD"), ("twin-vox", "vendor"), ("HemoPIC", "vendor")):
            for m in ("CBF", "CBV", "MTT", "mean"):
                d = np.array([r.get(f"{x} {m}", np.nan) - r.get(f"{y} {m}", np.nan) for r in rs])
                d = d[np.isfinite(d)]
                if d.size > 5:
                    tests[f"{x} - {y} {m}"] = {"diff": float(d.mean()), "p": float(wilcoxon(d).pvalue), "n": int(d.size),
                                              "n_better": int((d > 0).sum())}
        s["paired"] = tests
        summ[subset] = s
    (od / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
