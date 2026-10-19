from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

'''
export FREESURFER_HOME=<path to your freesurfer install>
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
export FS_LICENSE=<path to your license.txt>
'''

'''
Run:
python preprocess/Segmentation/run_synthseg.py \
  /path/to/ISLES2017_Training \
  /path/to/segmentation_root
'''

RIGHT_TO_LEFT_LABEL = {
    41: 2,
    42: 3,
    43: 4,
    44: 5,
    46: 7,
    47: 8,
    49: 10,
    50: 11,
    51: 12,
    52: 13,
    53: 17,
    54: 18,
    58: 26,
    60: 28,
}


def require_env(name: str) -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    raise RuntimeError(f"Missing required environment variable: {name}")


def require_executable(name: str) -> str:
    p = shutil.which(name)
    if p:
        return p
    raise RuntimeError(f"Cannot find executable in PATH: {name}")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_nonempty_file(p: Path) -> bool:
    if not p.exists():
        return False
    if not p.is_file():
        return False
    return p.stat().st_size > 0


def iter_subject_dirs(training_root: Path):
    for d in sorted(training_root.glob("training_*")):
        if d.is_dir():
            yield d


def run_synthseg_cli(
    input_image: Path,
    seg_out: Path,
    resample_out: Path,
    threads: int,
) -> None:
    cmd = [
        "mri_synthseg",
        "--i", str(input_image),
        "--o", str(seg_out),
        "--resample", str(resample_out),
        "--robust",
        "--autocrop",
        "--threads", str(int(threads)),
    ]
    subprocess.run(cmd, check=True)


def load_nifti(path: Path):
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    data = data.astype(np.int32)
    return data, img


def save_like(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    out = nib.Nifti1Image(data.astype(np.int16), affine=ref_img.affine, header=ref_img.header)
    out.set_qform(ref_img.get_qform(), code=int(ref_img.header.get("qform_code", 1)))
    out.set_sform(ref_img.get_sform(), code=int(ref_img.header.get("sform_code", 1)))
    nib.save(out, str(out_path))


def merge_right_to_left(seg: np.ndarray) -> np.ndarray:
    out = seg.copy()
    for r_lab, l_lab in RIGHT_TO_LEFT_LABEL.items():
        out[out == int(r_lab)] = int(l_lab)
    return out


def main() -> None:
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 run_synthseg.py <data_root> <segmentation_root> [target_subject] [max_threads]"
        )

    training_root = Path(sys.argv[1]).expanduser().resolve()
    out_root = Path(sys.argv[2]).expanduser().resolve()

    target_subject = sys.argv[3] if len(sys.argv) >= 4 else ""

    max_threads = int(sys.argv[4]) if len(sys.argv) >= 5 else 8

    require_env("FREESURFER_HOME")

    require_env("FS_LICENSE")

    require_executable("mri_synthseg")

    if not training_root.exists():
        raise RuntimeError(f"Training root not found: {training_root}")

    ensure_dir(out_root)
    cpu_threads = os.cpu_count() or 4
    threads = max(1, min(int(cpu_threads), int(max_threads)))

    processed = 0
    skipped = 0
    found_target = 0

    for subj_dir in iter_subject_dirs(training_root):
        if target_subject and subj_dir.name != target_subject:
            continue

        input_image = subj_dir / "MR_ADC.nii"

        if not input_image.exists():
            continue

        found_target += 1

        subj_out = ensure_dir(out_root / subj_dir.name / "synthseg")

        seg_out = subj_out / "seg_synthseg.nii.gz"
        resample_out = subj_out / "input_resampled.nii.gz"

        sym_out = subj_out / "seg_synthseg_sym.nii.gz"

        if is_nonempty_file(seg_out) and is_nonempty_file(resample_out) and is_nonempty_file(sym_out):
            skipped += 1
            continue

        if (not is_nonempty_file(seg_out)) or (not is_nonempty_file(resample_out)):
            run_synthseg_cli(input_image, seg_out, resample_out, threads)

        seg, ref_img = load_nifti(seg_out)

        seg_sym = merge_right_to_left(seg)

        save_like(seg_sym, ref_img, sym_out)

        processed += 1

    if target_subject and found_target == 0:
        raise RuntimeError("Target subject not found or MR_ADC.nii missing")

    print("Done")
    print("Processed", processed)
    print("Skipped", skipped)


if __name__ == "__main__":
    main()