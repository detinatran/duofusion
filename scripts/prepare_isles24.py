"""ISLES'24 training set in the CTP grid, as external/HemoPIC_plus/scripts/prepare_isles24.py,
plus what duofusion needs: the full SynthSeg labels (hemispheres, structures), the NCCT
resampled to the CTP grid and every scalar clinical field of the parquet row.

python scripts/prepare_isles24.py <hf_dir> <out_root> --freesurfer <FREESURFER_HOME> [--workers 3]

Writes, per subject, the files listed in duofusion/data.py. The HemoPIC files
(CTC, seg4, OT, vendor maps) are identical to the HemoPIC_plus preparation, so the
HemoPIC+ fits of HemoPIC_plus run on this folder unchanged.
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import SimpleITK as sitk  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "hp_prepare", ROOT / "external" / "HemoPIC_plus" / "scripts" / "prepare_isles24.py")
hp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hp)


def scalars(r):
    return {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None}


def prepare(r, num, d, fs_home):
    d.mkdir(parents=True, exist_ok=True)
    (d / "hemopic_seg4").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k in ("ncct", "ctp", "lesion_mask", *hp.MAPS):
            (tmp / f"{k}.nii.gz").write_bytes(r[k]["bytes"])
        ctp_nib = nib.load(str(tmp / "ctp.nii.gz"))
        ctp = np.asarray(ctp_nib.get_fdata(), np.float32)
        nib.save(nib.Nifti1Image(np.clip(ctp.mean(-1), -100, 200).astype(np.float32), ctp_nib.affine),
                 str(tmp / "ctp_mean.nii.gz"))
        ref = sitk.ReadImage(str(tmp / "ctp_mean.nii.gz"), sitk.sitkFloat32)
        ncct = sitk.ReadImage(str(tmp / "ncct.nii.gz"))
        tx, metric = hp.register(hp.clipped(ncct), ref)
        if metric > hp.MIN_MI:
            raise RuntimeError(f"registration failed (mutual information {metric:.3f})")
        inv = tx.GetInverse()
        cmd = (f'export FREESURFER_HOME="{fs_home}"; source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null 2>&1; '
               f'mri_synthseg --i "{tmp / "ncct.nii.gz"}" --o "{tmp / "seg.nii.gz"}" --robust --ct --threads 2')
        subprocess.run(["bash", "-c", cmd], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        seg30 = hp.sitk_to_nib(hp.to_ctp(sitk.ReadImage(str(tmp / "seg.nii.gz")), ref, inv, nearest=True))
        seg30 = np.rint(seg30).astype(np.int16)
        fov = np.all(ctp > -1000, axis=-1)
        seg30[~fov] = 0
        seg = hp.seg30_to_seg4(seg30)
        les = hp.sitk_to_nib(hp.to_ctp(sitk.ReadImage(str(tmp / "lesion_mask.nii.gz")), ref, inv)) >= 0.5
        ncct_ctp = hp.sitk_to_nib(hp.to_ctp(ncct, ref, inv)).astype(np.float32)
        brain = seg > 0
        nb = hp.baseline_frames(ctp, brain & (seg != 3))
        ctc = (ctp - ctp[..., :nb].mean(-1, keepdims=True)).astype(np.float32)
        ctc[~fov] = 0.0
        hdr = ctp_nib.header.copy()
        hdr.set_xyzt_units("mm", "sec")
        zooms = list(hdr.get_zooms())
        zooms[3] = 1.0
        hdr.set_zooms(zooms)
        aff = ctp_nib.affine
        nib.save(nib.Nifti1Image(ctc, aff, hdr), str(d / "CTC_from_MR_4DPWI.nii.gz"))
        nib.save(nib.Nifti1Image(seg, aff), str(d / "hemopic_seg4" / "seg4_label_ref24.nii.gz"))
        nib.save(nib.Nifti1Image(seg30, aff), str(d / "synthseg.nii.gz"))
        nib.save(nib.Nifti1Image(ncct_ctp, aff), str(d / "ncct.nii.gz"))
        nib.save(nib.Nifti1Image(les.astype(np.uint8), aff), str(d / "OT.nii.gz"))
        for k, name in hp.MAPS.items():
            v = hp.sitk_to_nib(hp.to_ctp(sitk.ReadImage(str(tmp / f"{k}.nii.gz")), ref, inv)).astype(np.float32)
            nib.save(nib.Nifti1Image(v, aff), str(d / f"{name}.nii.gz"))
        info = {**scalars(r), "registration_metric": metric, "transform_ctp_to_ncct": list(tx.GetParameters()),
                "baseline_frames": nb, "frames": int(ctp.shape[-1]), "dt_seconds": 1.0,
                "brain_voxels": int(brain.sum()), "lesion_voxels": int(les.sum()),
                "lesion_voxels_in_brain": int((les & brain).sum())}
        (d / "prep_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf8")
    return {"subject": num, **{k: info[k] for k in ("registration_metric", "baseline_frames", "brain_voxels",
                                                    "lesion_voxels_in_brain")}}


def work(args):
    path, out_root, fs_home, guard, min_free_gb = args
    free = shutil.disk_usage(guard).free / 2**30
    if free < min_free_gb:
        return {"file": Path(path).name, "skipped": f"only {free:.1f} GB free on {guard}"}
    r = pq.read_table(path).to_pylist()[0]
    num = int(r["subject_id"].split("stroke")[1])
    d = Path(out_root) / f"training_{num}"
    if (d / "prep_info.json").exists() and (d / "synthseg.nii.gz").exists():
        return {"subject": num, "skipped": "done"}
    try:
        return prepare(r, num, d, fs_home)
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(d, ignore_errors=True)
        return {"subject": num, "error": repr(e)[:300]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hf_dir", type=Path)
    p.add_argument("out_root", type=Path)
    p.add_argument("--freesurfer", required=True)
    p.add_argument("--workers", type=int, default=3, help="SynthSeg (robust) needs about 8 GB per worker")
    p.add_argument("--limit", type=int, default=0, help="only the first n subjects (testing)")
    p.add_argument("--guard-path", default=None, help="disk whose free space is checked before every subject")
    p.add_argument("--min-free-gb", type=float, default=20.0)
    a = p.parse_args()
    files = sorted(a.hf_dir.glob("train-*.parquet"))
    if a.limit:
        files = files[:a.limit]
    a.out_root.mkdir(parents=True, exist_ok=True)
    guard = a.guard_path if a.guard_path and Path(a.guard_path).exists() else str(a.out_root)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(work, [(f, a.out_root, a.freesurfer, guard, a.min_free_gb) for f in files]):
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    sys.exit(main())
