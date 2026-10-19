"""Timing maps (arrival delay, Tmax) and transit-time heterogeneity: lesion AUC.

python scripts/timing_maps.py --sources full full_aif full_tuned --npde full_aif__npde full__npde --workers 8

The final infarct (OT) of ISLES 2017 is usually best predicted by how late the
bolus arrives (Tmax), which the CBF / CBV / MTT maps do not show. The model
already fits a bolus delay per ROI ([C2]); this script turns it into maps and
scores every map with the lesion ROC AUC per patient (later / larger = lesion),
next to two model-free references computed from the same data:
  TTP          time to peak of each voxel (lightly smoothed curves);
  oSVD Tmax    time of the residue maximum of oSVD with the automatic AIF
               (patients with an accepted AIF only).
Model maps, for every results folder in --sources:
  delay        the fitted ROI delay;
  arrival      delay + mean of the [C2] dispersion kernel;
  delay+TTP    ROI delay plus each voxel's TTP offset from its ROI median;
  MTT          the model's own MTT map (for reference).
Neural PDE folders (--npde): CTH = CV x MTT and the CV of the transit times per ROI.
Writes <out_root>/timing/{per_patient.csv, summary.json} and prints a table.
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from hemopic_plus.aif import select_aif  # noqa: E402
from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, load_nifti, resolve_dt, squeeze_ctc, subject_id  # noqa: E402
from hemopic_plus.osvd import osvd_maps  # noqa: E402
from hemopic_plus.signal import positive_excursion  # noqa: E402


def auc_high(vol, brain, ot):
    ok = brain & np.isfinite(vol)
    if ot[ok].sum() == 0 or (~ot[ok]).sum() == 0:
        return np.nan
    return roc_auc(ot[ok].astype(np.uint8), vol[ok].astype(np.float64))


def roi_map(lab, values):
    out = np.full(lab.shape, np.nan)
    for rid, v in values.items():
        out[lab == rid] = v
    return out


def work(args):
    pid, cfg, out, sources, npde = args
    sub = subject_id(pid)
    base = cfg.resolve("data_root") / sub
    img, raw = load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))
    ctc = squeeze_ctc(raw)
    seg = load_nifti(base / SEG4_REL)[1].astype(int)
    ot = load_nifti(find_nii(base, "OT"))[1] > 0
    dt, _ = resolve_dt(cfg, img, pid)
    t = np.arange(ctc.shape[3]) * dt
    brain = (seg > 0) & np.all(np.isfinite(ctc), axis=3)
    curves = positive_excursion(ctc[brain].astype(np.float64), cfg.baseline_frames)
    ttp = np.full(seg.shape, np.nan)
    ttp[brain] = np.argmax(gaussian_filter1d(curves, 1.0, axis=1), axis=1) * dt
    res = {"patient": pid, "TTP": auc_high(ttp, brain, ot)}
    aif = select_aif(ctc, brain, t, cfg, seed=int(cfg.seed) + pid)
    if aif is not None:
        a = positive_excursion(aif["aif"][None, :], cfg.baseline_frames)[0]
        *_, tmax = osvd_maps(curves, a, dt, k_h=cfg.k_h, rho=cfg.rho_brain, return_tmax=True)
        vol = np.full(seg.shape, np.nan)
        vol[brain] = tmax
        res["oSVD Tmax"] = auc_high(vol, brain, ot)
    for src in sources:
        fd = out / src / sub
        if not (fd / "roi_fit_summary.csv").exists():
            continue
        lab = nib.load(str(fd / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
        rows = {int(r["roi_id"]): r for r in csv.DictReader(open(fd / "roi_fit_summary.csv", encoding="utf8"))}
        ok = {k: r for k, r in rows.items() if r["success"] == "True"}
        delay = roi_map(lab, {k: float(r["delay_fit"]) for k, r in ok.items()})
        arrival = roi_map(lab, {k: float(r["delay_fit"]) + float(r["dispersion_fit"]) for k, r in ok.items()})
        off = np.full(seg.shape, np.nan)
        for k in ok:
            m = (lab == k) & np.isfinite(ttp)
            if m.any():
                off[m] = ttp[m] - np.median(ttp[m])
        res[f"{src}: delay"] = auc_high(delay, brain, ot)
        res[f"{src}: arrival"] = auc_high(arrival, brain, ot)
        res[f"{src}: delay+TTP"] = auc_high(delay + off, brain, ot)
        mtt = nib.load(str(fd / "HemoPIC_MTT.nii.gz")).get_fdata()
        res[f"{src}: MTT"] = auc_high(mtt, brain, ot)
    for src in npde:
        fd = out / src / sub
        f = fd / "npde_theta.npz"
        if not f.exists():
            continue
        import torch
        from hemopic_plus.uq.data import load_subject
        from hemopic_plus.uq.torch_model import DTYPE, derived
        z = np.load(f)
        b = load_subject(out / src.split("__")[0] / sub)
        with torch.no_grad():
            mtt_roi = derived(b, torch.as_tensor(z["theta"][:, :7], dtype=DTYPE))["MTT"].numpy()
        lab = nib.load(str(fd / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
        cv = {int(r): float(c) for r, c in zip(z["roi_ids"], z["cv_npde"])}
        cth = {int(r): float(c) * float(m) for r, c, m in zip(z["roi_ids"], z["cv_npde"], mtt_roi)}
        res[f"{src}: CTH"] = auc_high(roi_map(lab, cth), brain, ot)
        res[f"{src}: CV"] = auc_high(roi_map(lab, cv), brain, ot)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/full_aif.yaml"),
                   help="for data root, TR and the AIF settings of the oSVD reference")
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--sources", nargs="*", default=["full", "full_aif", "full_tuned"])
    p.add_argument("--npde", nargs="*", default=["full_aif__npde", "full__npde"])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out = cfg.resolve("out_root")
    pids = list_patients(cfg.resolve("data_root"))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(work, [(pid, cfg, out, a.sources, a.npde) for pid in sorted(pids)]))
    od = out / "timing"
    od.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r if k != "patient"))
    with open(od / "per_patient.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["patient", *keys])
        w.writeheader()
        w.writerows(rows)
    val = {k: {r["patient"]: r[k] for r in rows if k in r and np.isfinite(r[k])} for k in keys}
    summ = {k: {"auc": float(np.mean(list(v.values()))), "n": len(v)} for k, v in val.items()}

    def paired(x, y):
        common = sorted(set(val[x]) & set(val[y]))
        d = np.array([val[x][p] - val[y][p] for p in common])
        return {"diff": float(d.mean()), "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, "n": len(common)}

    tests = {}
    for k in keys:
        if k in ("TTP", "oSVD Tmax"):
            continue
        for ref in ("oSVD Tmax", "TTP"):
            if ref in val:
                tests[f"{k} - {ref}"] = paired(k, ref)
    for src in a.sources:
        if f"{src}: delay+TTP" in val and f"{src}: MTT" in val:
            tests[f"{src}: delay+TTP - {src}: MTT"] = paired(f"{src}: delay+TTP", f"{src}: MTT")
    (od / "summary.json").write_text(json.dumps({"auc": summ, "paired": tests}, indent=2), encoding="utf8")
    print("lesion ROC AUC per patient (larger value = lesion; mean over patients)")
    for k, s in summ.items():
        print(f"  {k:32s} {s['auc']:.3f}  (n={s['n']})")
    print("paired differences (Wilcoxon)")
    for k, s in tests.items():
        print(f"  {k:52s} {s['diff']:+.3f}  p={s['p']:.2g}  n={s['n']}")


if __name__ == "__main__":
    main()
