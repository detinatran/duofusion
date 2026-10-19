"""In-plane spatial smoothing of a prepared perfusion set (scripts/prepare_isles24.py).

python scripts/smooth_external.py <ISLES2024_Training> <ISLES2024_Training_s3> [--sigma 3] [--workers 12]

CT perfusion is much noisier per voxel than the MR set. Every subject folder is
copied as links (as files where links are not supported, e.g. Google Drive)
except the tracer concentration, which is smoothed within the
brain mask (seg4 > 0) by a Gaussian of --sigma voxels in the slice plane
(normalised convolution, no smoothing across slices or in time). Every method is
then run on the smoothed copy, so all of them see the same input.
"""
import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

CTC = "CTC_from_MR_4DPWI.nii.gz"


def work(args):
    src, dst, sigma = args
    out = dst / src.name
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != CTC and not (out / f.name).exists():
            try:
                os.symlink(f.resolve(), out / f.name)
            except OSError:
                if f.is_dir():
                    shutil.copytree(f, out / f.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(f, out / f.name)
    if (out / CTC).exists():
        return src.name
    img = nib.load(str(src / CTC))
    x = np.asarray(img.get_fdata(), np.float32)
    brain = nib.load(str(src / "hemopic_seg4" / "seg4_label_ref24.nii.gz")).get_fdata() > 0
    num = gaussian_filter(np.where(brain[..., None], x, 0.0), sigma=(sigma, sigma, 0, 0))
    den = gaussian_filter(brain.astype(np.float32), sigma=(sigma, sigma, 0))[..., None]
    y = np.where(brain[..., None], num / np.maximum(den, 1e-6), x).astype(np.float32)
    nib.save(nib.Nifti1Image(y, img.affine, img.header), str(out / CTC))
    return src.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--sigma", type=float, default=3.0, help="in-plane Gaussian, in voxels")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    subjects = sorted(d for d in a.src.glob("training_*") if (d / "prep_info.json").exists())
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        n = sum(1 for _ in ex.map(work, [(d, a.dst, a.sigma) for d in subjects]))
    print(f"smoothed {n} subjects -> {a.dst}")


if __name__ == "__main__":
    main()
