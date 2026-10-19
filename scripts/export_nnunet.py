"""nnU-Net v2 raw dataset of ISLES'24 with the vendor maps and the HemoPIC+ maps as channels.

python scripts/export_nnunet.py <ISLES2024_Training> <twin_out_s3> <out_dir> --folds scripts/learned_folds24.json

Channels (all in the CTP grid, outside the brain set to 0):
  0 NCCT (CT normalisation in nnU-Net)
  1-4 vendor CBF, CBV, MTT, Tmax          (vendor dataset = channels 0-4)
  5-8 HemoPIC+ CBF, CBV, MTT, delay         (the voxel-fit maps of the paper: CBV = ID-CBV,
                                           MTT = voxel MTT, CBF = 60 CBV / MTT; delay = voxel delay)
CBF, CBV and MTT of both sources are divided by their brain median (as for the lesion AUC),
Tmax and delay are kept in seconds; every perfusion map is clipped to a robust range.
Label: the follow-up infarct (OT). Writes Dataset_ISLES24/{imagesTr,labelsTr,dataset.json,
splits_final.json} under <out_dir> (5 folds = the patient folds of the duofusion evaluation).
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def load(p):
    img = nib.load(str(p))
    return img, np.asarray(img.get_fdata(), np.float32)


def rel(v, brain):
    ok = brain & np.isfinite(v) & (v > 0)
    med = float(np.median(v[ok])) if np.any(ok) else 1.0
    return np.clip(np.where(ok, v / max(med, 1e-6), 0.0), 0, 10)


def secs(v, brain, lo=-10, hi=30):
    return np.clip(np.where(brain & np.isfinite(v), v, 0.0), lo, hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=Path)
    p.add_argument("twin_out", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--name", default="Dataset501_ISLES24Twin")
    a = p.parse_args()
    folds = {int(k): v for k, v in json.loads(a.folds.read_text()).items()}
    root = a.out_dir / a.name
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    vox = a.twin_out / "isles24_full_aif__vox_tank"
    idc = a.twin_out / "isles24_full_aif_idvox_osvd_nc"
    cases = []
    for n in sorted(folds):
        d = a.data_root / f"training_{n}"
        need = [d / "ncct.nii.gz", d / "OT.nii.gz", d / "hemopic_seg4" / "seg4_label_ref24.nii.gz",
                vox / f"training_{n}" / "VOX_MTT.nii.gz", vox / f"training_{n}" / "VOX_delay.nii.gz",
                idc / f"training_{n}" / "HemoPIC_CBV.nii.gz"] + [d / f"MR_{m}.nii.gz" for m in ("rCBF", "rCBV", "MTT", "Tmax")]
        if not all(f.exists() for f in need):
            print("skip", n, [f.name for f in need if not f.exists()])
            continue
        ref, seg = load(d / "hemopic_seg4" / "seg4_label_ref24.nii.gz")
        brain = seg > 0
        aff = ref.affine
        ncct = load(d / "ncct.nii.gz")[1]
        vcbf, vcbv, vmtt, vtmax = (load(d / f"MR_{m}.nii.gz")[1] for m in ("rCBF", "rCBV", "MTT", "Tmax"))
        mtt = load(vox / f"training_{n}" / "VOX_MTT.nii.gz")[1]
        delay = load(vox / f"training_{n}" / "VOX_delay.nii.gz")[1]
        cbv = load(idc / f"training_{n}" / "HemoPIC_CBV.nii.gz")[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            cbf = np.where(np.isfinite(mtt) & (mtt > 0), 60.0 * cbv / mtt, np.nan)
        chans = [np.nan_to_num(ncct), rel(vcbf, brain), rel(vcbv, brain), rel(vmtt, brain), secs(vtmax, brain, 0, 30),
                 rel(cbf, brain), rel(cbv, brain), rel(mtt, brain), secs(delay, brain)]
        cid = f"isles24_{n:03d}"
        for c, v in enumerate(chans):
            nib.save(nib.Nifti1Image(v.astype(np.float32), aff), str(root / "imagesTr" / f"{cid}_{c:04d}.nii.gz"))
        ot = (load(d / "OT.nii.gz")[1] > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(ot, aff), str(root / "labelsTr" / f"{cid}.nii.gz"))
        cases.append((cid, folds[n]))
        print(cid, "ok", flush=True)
    names = ["CT", "vendor_CBF", "vendor_CBV", "vendor_MTT", "vendor_Tmax", "twin_CBF", "twin_CBV", "twin_MTT", "twin_delay"]
    ds = {"channel_names": {str(i): ("CT" if i == 0 else "zscore") for i in range(len(names))},
          "channel_descriptions": {str(i): n for i, n in enumerate(names)},
          "labels": {"background": 0, "infarct": 1}, "numTraining": len(cases), "file_ending": ".nii.gz"}
    (root / "dataset.json").write_text(json.dumps(ds, indent=1))
    splits = [{"train": [c for c, f in cases if f != k], "val": [c for c, f in cases if f == k]} for k in range(5)]
    (root / "splits_final.json").write_text(json.dumps(splits, indent=1))
    print(len(cases), "cases ->", root)


if __name__ == "__main__":
    main()
