"""Write HemoPIC's four tissue classes (hemopic_seg4/seg4_label_ref24.nii.gz) from the
SynthSeg-coded labels of a prepared cohort, so that the HemoPIC+ fits of HemoPIC_plus run
on it (needed for ISLES 2018, whose preparation writes pseudo labels only).

python scripts/add_seg4.py <data_root>
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from duofusion.labels import BRAINSTEM, CSF, WM  # noqa: E402


def main():
    root = Path(sys.argv[1])
    n = 0
    for d in sorted(root.glob("training_*")):
        out = d / "hemopic_seg4" / "seg4_label_ref24.nii.gz"
        if out.exists() or not (d / "synthseg.nii.gz").exists():
            continue
        img = nib.load(str(d / "synthseg.nii.gz"))
        seg = np.rint(img.get_fdata()).astype(np.int32)
        s4 = np.zeros(seg.shape, np.uint8)
        wm, csf, bs = np.isin(seg, list(WM)), np.isin(seg, list(CSF)), np.isin(seg, list(BRAINSTEM))
        s4[wm], s4[csf], s4[bs] = 2, 3, 4
        s4[(seg > 0) & ~(wm | csf | bs)] = 1
        out.parent.mkdir(exist_ok=True)
        nib.save(nib.Nifti1Image(s4, img.affine), str(out))
        n += 1
    print(f"{n} seg4 files written")


if __name__ == "__main__":
    main()
