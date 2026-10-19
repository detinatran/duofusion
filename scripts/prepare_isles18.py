"""ISLES 2018 training slabs (CT perfusion, core on DWI acquired soon after the CTP) in the
layout of duofusion/data.py, so that the duofusion scripts run unchanged.

python scripts/prepare_isles18.py <TRAINING dir> <out_root> [--workers 8]

The slabs (2-8 slices of 5 mm) are too thin for SynthSeg, so the anatomy is built from
the slab itself and written as SynthSeg-coded pseudo labels (synthseg.nii.gz):
  brain   vendor CBF > 0 (the challenge maps are skull-stripped) and 0 < CT < 100 HU
  CSF     brain with CT < 15 HU                                   -> label 24 (excluded)
  grey    parenchyma at or above the case's median CT (contrast-enhanced
          CTP average, grey matter enhances more)                 -> 3 / 42
  white   parenchyma below the median                             -> 2 / 41
  side    from the midsagittal plane of duofusion.symmetry.Mirror (in-plane registration)
The CT of the challenge is the temporal average of the CTP; it is stored as ncct.nii.gz.
prep_info.json holds the case, the patient (Train_<p>_<slab>) and quality numbers; the
pilot uses "patient" to keep slabs of one patient in the same fold.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from duofusion.symmetry import Mirror  # noqa: E402

MODS = {"CT": "ncct", "CT_CBF": "MR_rCBF", "CT_CBV": "MR_rCBV", "CT_MTT": "MR_MTT", "CT_Tmax": "MR_Tmax", "OT": "OT"}


def find(case, mod):
    hits = [p for p in case.glob(f"*.{mod}.*/*.nii") if re.search(rf"\.O\.{mod}\.\d+$", p.parent.name)]
    return hits[0] if hits else None


def patient_id(case):
    for j in case.glob("*/*.json"):
        m = re.search(r"Train_(\d+)_([A-Z])", j.read_text(encoding="utf8", errors="ignore"))
        if m:
            return int(m.group(1)), m.group(2)
    return None, None


def baseline_frames(ctp, brain):
    m = ctp[brain].mean(axis=0)
    rise = m - m[:3].mean()
    return max(2, int(np.argmax(rise > 0.1 * rise.max())) - 1)


def work(args):
    case, out_root = args
    num = int(case.name.split("_")[1])
    d = Path(out_root) / f"training_{num}"
    if (d / "prep_info.json").exists():
        return {"case": num, "skipped": "done"}
    try:
        imgs = {k: nib.load(str(find(case, k))) for k in MODS}
        aff = imgs["CT"].affine
        vol = {k: np.asarray(v.dataobj, np.float32) for k, v in imgs.items()}
        ct, cbf = vol["CT"], vol["CT_CBF"]
        brain = (cbf > 0) & (ct > 0) & (ct < 100)
        csf = brain & (ct < 15)
        par = brain & ~csf
        med = float(np.median(ct[par])) if np.any(par) else 30.0
        mir = Mirror(aff, np.clip(ct, 0, 80), brain)
        side = mir.side(brain)
        seg = np.zeros(ct.shape, np.int16)
        grey, white = par & (ct >= med), par & (ct < med)
        seg[grey & (side == 1)], seg[grey & (side == 2)] = 3, 42
        seg[white & (side == 1)], seg[white & (side == 2)] = 2, 41
        seg[csf] = 24
        d.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(seg, aff), str(d / "synthseg.nii.gz"))
        for k, name in MODS.items():
            v = (vol[k] > 0).astype(np.uint8) if k == "OT" else vol[k]
            nib.save(nib.Nifti1Image(v, aff), str(d / f"{name}.nii.gz"))
        p4 = nib.load(str(find(case, "CT_4DPWI")))
        ctp = np.asarray(p4.dataobj, np.float32)
        nb = baseline_frames(ctp, brain)
        ctc = (ctp - ctp[..., :nb].mean(-1, keepdims=True)).astype(np.float32)
        ctc[~brain] = 0.0
        hdr = p4.header.copy()
        hdr.set_xyzt_units("mm", "sec")
        nib.save(nib.Nifti1Image(ctc, p4.affine, hdr), str(d / "CTC_from_MR_4DPWI.nii.gz"))
        pat, slab = patient_id(case)
        ot = vol["OT"] > 0
        info = {"case": num, "patient": pat, "slab": slab, "slices": int(ct.shape[2]),
                "voxel_mm": [float(x) for x in np.sqrt((aff[:3, :3] ** 2).sum(0))],
                "frames": int(ctp.shape[-1]), "dt_seconds": float(p4.header.get_zooms()[3] or 1.0),
                "baseline_frames": nb, "mirror_metric": mir.metric, "mirror_planar": bool(mir.planar),
                "ct_median_parenchyma": med, "brain_voxels": int(brain.sum()),
                "side_balance": float((side == 1).sum() / max((side > 0).sum(), 1)),
                "lesion_voxels": int(ot.sum()), "lesion_in_brain": float((ot & brain).sum() / max(ot.sum(), 1))}
        (d / "prep_info.json").write_text(json.dumps(info, indent=1), encoding="utf8")
        return {k: info[k] for k in ("case", "patient", "slab", "slices", "mirror_metric", "side_balance", "lesion_in_brain")}
    except Exception as e:  # noqa: BLE001
        return {"case": num, "error": repr(e)[:300]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("training_dir", type=Path)
    p.add_argument("out_root", type=Path)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    cases = sorted(a.training_dir.glob("case_*"), key=lambda c: int(c.name.split("_")[1]))
    a.out_root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(work, [(c, a.out_root) for c in cases]):
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
