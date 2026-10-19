"""ISLES'24 training set (CT perfusion) in the layout of ISLES 2017, for the external test.

python scripts/prepare_isles24.py <hf_dir> <out_root> --freesurfer <FREESURFER_HOME> [--workers 8]

Input: the parquet files of the Hugging Face mirror of the ISLES'24 training set
(hugging-science/isles24-stroke, one subject per file: raw 4D CTP, NCCT, the
follow-up infarct mask and the vendor perfusion maps in NCCT space). The raw CTP
is motion corrected and resampled to 1 frame per second by the challenge.
For every subject, in the CTP grid (out_root/training_<n>/, n = the subject number):
  CTC_from_MR_4DPWI.nii.gz        tracer concentration, HU minus the pre-bolus
                                  baseline (CT enhancement is linear in the
                                  contrast concentration), dt = 1 s in the header;
  hemopic_seg4/seg4_label_ref24.nii.gz  HemoPIC's four tissue classes from SynthSeg
                                  (robust, CT mode) on the NCCT;
  OT.nii.gz                       final infarct (follow-up DWI mask, area fraction >= 0.5);
  MR_rCBF, MR_rCBV, MR_MTT, MR_Tmax .nii.gz  the vendor maps (reference row);
  prep_info.json                  registration and baseline details.
NCCT and the raw CTP do not share world coordinates, so the temporal mean of the
CTP is registered rigidly to the NCCT (Mattes mutual information, SimpleITK) and
every NCCT-space image is resampled into the CTP grid with the inverse transform.
A subject whose best registration stays below a mutual information of 0.5 is
left out (it is reported, and no folder is written).
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import SimpleITK as sitk  # noqa: E402

WM, CSF, BRAINSTEM = {2, 41}, {4, 5, 14, 24, 43, 44}, {16}
MAPS = {"cbf": "MR_rCBF", "cbv": "MR_rCBV", "mtt": "MR_MTT", "tmax": "MR_Tmax"}
MIN_MI = -0.5


def seg30_to_seg4(seg):
    """HemoPIC preprocess/Segmentation/hemopic_seg4.py: 1 GM, 2 WM, 3 CSF, 4 brain stem."""
    seg = seg.astype(np.int32)
    out = np.zeros(seg.shape, np.uint8)
    wm, csf, bs = np.isin(seg, list(WM)), np.isin(seg, list(CSF)), np.isin(seg, list(BRAINSTEM))
    out[wm], out[csf], out[bs] = 2, 3, 4
    out[(seg > 0) & ~(wm | csf | bs)] = 1
    return out


def register(fixed, moving):
    """Rigid registration from several starts (image centres or intensity moments,
    shifted along z by up to 40 mm since the CTP covers a thin slab); the start
    with the best final mutual information is kept."""
    best = None
    for mode in (sitk.CenteredTransformInitializerFilter.GEOMETRY, sitk.CenteredTransformInitializerFilter.MOMENTS):
        for dz in (0.0, -20.0, 20.0, -40.0, 40.0):
            init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), mode)
            init = sitk.Euler3DTransform(init)
            tr = list(init.GetTranslation())
            tr[2] += dz
            init.SetTranslation(tr)
            try:
                tx, metric = register_from(fixed, moving, init)
            except RuntimeError:
                continue
            if best is None or metric < best[1]:
                best = (tx, metric)
    if best is None:
        raise RuntimeError("registration failed from every start")
    return best


def register_from(fixed, moving, init):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.05, 1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, 300, relaxationFactor=0.6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SetInitialTransform(init, inPlace=False)
    tx = R.Execute(fixed, moving)
    return tx, float(R.GetMetricValue())


def clipped(img):
    return sitk.Clamp(sitk.Cast(img, sitk.sitkFloat32), sitk.sitkFloat32, -100.0, 200.0)


def to_ctp(img, ref, inv, nearest=False):
    return sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(img, sitk.sitkFloat32), ref, inv,
                                                sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear, 0.0))


def sitk_to_nib(arr_zyx):
    return np.transpose(arr_zyx, (2, 1, 0))


def baseline_frames(ctp, brain):
    """Frames before the bolus: the whole-brain mean stays below 10% of its rise."""
    m = ctp[brain].mean(axis=0)
    b0 = m[:3].mean()
    rise = m - b0
    k = int(np.argmax(rise > 0.1 * rise.max()))
    return max(2, k - 1)


def work(args):
    """One subject; a failure removes its folder so that the fits skip it."""
    path, out_root, fs_home = args
    r = pq.read_table(path).to_pylist()[0]
    num = int(r["subject_id"].split("stroke")[1])
    d = Path(out_root) / f"training_{num}"
    if (d / "prep_info.json").exists():
        return {"subject": num, "skipped": "done"}
    try:
        return prepare(r, num, d, fs_home)
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(d, ignore_errors=True)
        return {"subject": num, "error": repr(e)[:300]}


def prepare(r, num, d, fs_home):
    d.mkdir(parents=True, exist_ok=True)
    (d / "hemopic_seg4").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k in ("ncct", "ctp", "lesion_mask", *MAPS):
            (tmp / f"{k}.nii.gz").write_bytes(r[k]["bytes"])
        ctp_nib = nib.load(str(tmp / "ctp.nii.gz"))
        ctp = np.asarray(ctp_nib.get_fdata(), np.float32)
        mean_nib = nib.Nifti1Image(np.clip(ctp.mean(-1), -100, 200).astype(np.float32), ctp_nib.affine)
        nib.save(mean_nib, str(tmp / "ctp_mean.nii.gz"))
        ref = sitk.ReadImage(str(tmp / "ctp_mean.nii.gz"), sitk.sitkFloat32)
        ncct = sitk.ReadImage(str(tmp / "ncct.nii.gz"))
        tx, metric = register(clipped(ncct), ref)
        if metric > MIN_MI:
            raise RuntimeError(f"registration failed (mutual information {metric:.3f})")
        inv = tx.GetInverse()
        cmd = (f'export FREESURFER_HOME="{fs_home}"; source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null 2>&1; '
               f'mri_synthseg --i "{tmp / "ncct.nii.gz"}" --o "{tmp / "seg.nii.gz"}" --robust --ct --threads 2')
        subprocess.run(["bash", "-c", cmd], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        seg = seg30_to_seg4(sitk_to_nib(to_ctp(sitk.ReadImage(str(tmp / "seg.nii.gz")), ref, inv, nearest=True)))
        les = sitk_to_nib(to_ctp(sitk.ReadImage(str(tmp / "lesion_mask.nii.gz")), ref, inv)) >= 0.5
        fov = np.all(ctp > -1000, axis=-1)
        seg[~fov] = 0
        brain = seg > 0
        nb = baseline_frames(ctp, brain & (seg != 3))
        ctc = (ctp - ctp[..., :nb].mean(-1, keepdims=True)).astype(np.float32)
        ctc[~fov] = 0.0
        hdr = ctp_nib.header.copy()
        hdr.set_xyzt_units("mm", "sec")
        zooms = list(hdr.get_zooms())
        zooms[3] = 1.0
        hdr.set_zooms(zooms)
        nib.save(nib.Nifti1Image(ctc, ctp_nib.affine, hdr), str(d / "CTC_from_MR_4DPWI.nii.gz"))
        nib.save(nib.Nifti1Image(seg, ctp_nib.affine), str(d / "hemopic_seg4" / "seg4_label_ref24.nii.gz"))
        nib.save(nib.Nifti1Image(les.astype(np.uint8), ctp_nib.affine), str(d / "OT.nii.gz"))
        for k, name in MAPS.items():
            v = sitk_to_nib(to_ctp(sitk.ReadImage(str(tmp / f"{k}.nii.gz")), ref, inv)).astype(np.float32)
            nib.save(nib.Nifti1Image(v, ctp_nib.affine), str(d / f"{name}.nii.gz"))
        info = {"subject": r["subject_id"], "registration_metric": metric, "transform_ctp_to_ncct": list(tx.GetParameters()),
                "baseline_frames": nb, "frames": int(ctp.shape[-1]), "dt_seconds": 1.0,
                "brain_voxels": int(brain.sum()), "lesion_voxels": int(les.sum()), "lesion_voxels_in_brain": int((les & brain).sum()),
                "age": r.get("age"), "nihss_admission": r.get("nihss_admission"), "mrs_3month": r.get("mrs_3month")}
        (d / "prep_info.json").write_text(json.dumps(info, indent=1), encoding="utf8")
    return {"subject": num, **{k: info[k] for k in ("registration_metric", "baseline_frames", "brain_voxels", "lesion_voxels_in_brain")}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("hf_dir", type=Path)
    p.add_argument("out_root", type=Path)
    p.add_argument("--freesurfer", required=True)
    p.add_argument("--workers", type=int, default=4, help="SynthSeg (robust) needs several GB per worker")
    a = p.parse_args()
    files = sorted(a.hf_dir.glob("train-*.parquet"))
    a.out_root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(work, [(f, a.out_root, a.freesurfer) for f in files]):
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
