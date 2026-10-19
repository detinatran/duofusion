"""Lesion metrics beyond the AUC (hemopic_plus/lesion_metrics.py) for every method of
the paper's lesion table, on the ISLES 2017 patients with oSVD maps, and the
values of the maps in normal tissue.

python scripts/eval_lesion_metrics.py configs/full_aif.yaml [--workers 24]

Methods and maps as eval_voxel_maps.py / the paper (twin-vox: indicator-dilution
CBV, voxel MTT of the compartment fit, CBF = 60 CBV / MTT). Normal tissue: grey
(seg4 = 1) and white matter (seg4 = 2) outside the final infarct dilated by
3 voxels; per patient the median CBF, CBV and MTT, and the grey/white CBF ratio.
Writes <out_root>/lesion_metrics/{per_patient.csv, summary.json, physiology.csv}.
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
from scipy.ndimage import binary_dilation  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import DECONV, MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, subject_id  # noqa: E402
from hemopic_plus.lesion_metrics import map_metrics, paired_bootstrap, summarize  # noqa: E402

ID_CBV = "full_aif_idvox_osvd_nc"
FOLDERS = {"HemoPIC": "baseline", "HemoPIC+ (compartment)": "full_aif", "HemoPIC+ w/ PDE": "full_aif__pde",
           "HemoPIC+ w/ neural PDE": "full_aif__npde", "HemoPIC+ ID-CBV, oSVD texture": ID_CBV}
METHODS = ["ISLES deconvolution", "HemoPIC", "oSVD", "HemoPIC+ (compartment)", "HemoPIC+ w/ PDE", "HemoPIC+ w/ neural PDE",
           "HemoPIC+ ID-CBV, oSVD texture", "twin-vox"]
ID_CBV_NAME = "HemoPIC+ ID-CBV, oSVD texture"
PHYSIO = ["HemoPIC", "oSVD", "HemoPIC+ (compartment)", "HemoPIC+ w/ neural PDE", "twin-vox"]


def load(p):
    return nib.load(str(p)).get_fdata() if p is not None and Path(p).exists() else None


def method_maps(out, base, sub):
    maps = {"ISLES deconvolution": {m: load(find_nii(base, DECONV[m])) for m, _ in MAPS},
            "oSVD": {m: load(out / "osvd" / sub / f"oSVD_{m}.nii.gz") for m, _ in MAPS}}
    for name, folder in FOLDERS.items():
        maps[name] = {m: load(out / folder / sub / f"HemoPIC_{m}.nii.gz") for m, _ in MAPS}
    cbv, mtt = maps[ID_CBV_NAME]["CBV"], load(out / "full_aif__vox_tank" / sub / "VOX_MTT.nii.gz")
    if cbv is not None and mtt is not None:
        maps["twin-vox"] = {"CBV": cbv, "MTT": mtt,
                              "CBF": np.where(np.isfinite(mtt) & (mtt > 0), 60.0 * cbv / np.maximum(mtt, 1e-6), np.nan)}
    return maps


def work(args):
    pid, out, data_root = args
    sub = subject_id(pid)
    base = data_root / sub
    seg_img = nib.load(str(base / SEG4_REL))
    seg = np.asarray(seg_img.get_fdata()).astype(int)
    brain = seg > 0
    ot = load(find_nii(base, "OT")) > 0
    voxel_ml = float(np.prod(seg_img.header.get_zooms()[:3])) / 1000.0
    maps = method_maps(out, base, sub)
    rec = {"patient": pid}
    for meth, vols in maps.items():
        rec[meth] = {}
        for m, direction in MAPS:
            vol = vols.get(m)
            mm = map_metrics(vol, brain, ot, direction, voxel_ml)
            if mm is None:
                continue
            valid, score = _lesion_scores(vol, brain, direction)
            mm["auc"] = roc_auc(ot[valid], score) if score is not None else np.nan
            rec[meth][m] = mm
    normal = ~binary_dilation(ot, iterations=3)
    phys = {"patient": pid}
    for meth in PHYSIO:
        for m, _ in MAPS:
            vol = maps.get(meth, {}).get(m)
            if vol is None:
                continue
            for lab, t in ((1, "GM"), (2, "WM")):
                sel = (seg == lab) & normal & np.isfinite(vol) & (vol > 0)
                phys[f"{meth} {m} {t}"] = float(np.median(vol[sel])) if sel.any() else np.nan
        if f"{meth} CBF GM" in phys:
            phys[f"{meth} CBF GM/WM"] = phys[f"{meth} CBF GM"] / phys[f"{meth} CBF WM"]
    return rec, phys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--patients", type=int, nargs="*")
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out, data_root = cfg.resolve("out_root"), cfg.resolve("data_root")
    pids = a.patients or sorted(int(d.name.split("_")[1]) for d in (out / "osvd").glob("training_*")
                                if (d / "oSVD_CBF.nii.gz").exists())
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(work, [(pid, out, data_root) for pid in pids]))
    records, phys = [r for r, _ in res], [q for _, q in res]
    rows, summ = summarize(records, METHODS, MAPS)
    for r in rows:
        for meth in METHODS:
            r[f"{meth} mean AUC"] = np.nanmean([r.get(f"{meth} {m} AUC", np.nan) for m, _ in MAPS])
            r[f"{meth} mean AP"] = np.nanmean([r.get(f"{meth} {m} AP", np.nan) for m, _ in MAPS])
    summ["paired, twin-vox minus"] = {}
    for ref in METHODS[:-1]:
        summ["paired, twin-vox minus"][ref] = {
            k: paired_bootstrap(rows, f"twin-vox {k}", f"{ref} {k}")
            for k in ("mean AUC", "mean AP", "CBF Dice CV", "MTT Dice CV", "CBF AP", "MTT AP")}
    keys = sorted({k for p_ in phys for k in p_ if k != "patient"})
    summ["physiology, median over patients (normal GM / WM)"] = {
        k: [float(np.nanmedian([p_.get(k, np.nan) for p_ in phys])),
            float(np.nanpercentile([p_.get(k, np.nan) for p_ in phys], 25)),
            float(np.nanpercentile([p_.get(k, np.nan) for p_ in phys], 75))] for k in keys}
    od = out / "lesion_metrics"
    od.mkdir(parents=True, exist_ok=True)
    rkeys = ["patient"] + sorted({k for r in rows for k in r if k != "patient"})
    with open(od / "per_patient.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=rkeys)
        w.writeheader()
        w.writerows(rows)
    with open(od / "physiology.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["patient"] + keys)
        w.writeheader()
        w.writerows(phys)
    (od / "summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf8")
    print(f"lesion metrics on {len(rows)} patients (mean [95% bootstrap CI])")
    for meth, sm in summ["methods"].items():
        line = f"  {meth:30s} AUC {sm['mean AUC'][0]:.3f} [{sm['mean AUC'][1]:.3f}, {sm['mean AUC'][2]:.3f}]  AP {sm['mean AP'][0]:.3f}"
        for m in ("CBF", "MTT"):
            if m in sm:
                line += f"  {m} Dice {sm[m]['Dice (CV threshold)'][0]:.3f} |dV| {sm[m]['volume abs error ml (CV threshold)'][0]:.1f}ml"
        print(line)
    for ref, d in summ["paired, twin-vox minus"].items():
        if d["mean AUC"]:
            print(f"  twin-vox - {ref:28s} AUC {d['mean AUC']['diff']:+.3f} {d['mean AUC']['ci']}  AP {d['mean AP']['diff']:+.3f}")
    for k in keys:
        if k.startswith("twin-vox") or k.startswith("oSVD"):
            print(f"  {k:30s} {summ['physiology, median over patients (normal GM / WM)'][k]}")


if __name__ == "__main__":
    main()
