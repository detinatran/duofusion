"""Subjects prepared by scripts/prepare_isles24.py (CTP grid, one folder per subject).

<root>/training_<n>/
  CTC_from_MR_4DPWI.nii.gz   tracer concentration (HU minus baseline), dt = 1 s
  synthseg.nii.gz            SynthSeg labels (FreeSurfer LUT) on the NCCT, in the CTP grid
  ncct.nii.gz                NCCT in the CTP grid
  OT.nii.gz                  follow-up infarct mask
  MR_rCBF, MR_rCBV, MR_MTT, MR_Tmax .nii.gz   vendor maps
  hemopic_seg4/seg4_label_ref24.nii.gz         HemoPIC tissue classes (for the HemoPIC+ fits)
  prep_info.json
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from .labels import hemisphere_map, tissue_map

VENDOR = {"CBF": "MR_rCBF", "CBV": "MR_rCBV", "MTT": "MR_MTT", "Tmax": "MR_Tmax"}


def _load(path):
    img = nib.load(str(path))
    return img, np.asarray(img.get_fdata(), np.float32)


def list_subjects(root):
    """Subject numbers with a finished preparation, sorted."""
    nums = []
    for d in Path(root).glob("training_*"):
        m = re.fullmatch(r"training_(\d+)", d.name)
        if m and (d / "prep_info.json").exists() and (d / "synthseg.nii.gz").exists():
            nums.append(int(m.group(1)))
    return sorted(nums)


@dataclass
class Subject:
    num: int
    folder: Path
    affine: np.ndarray
    seg: np.ndarray            # SynthSeg labels
    tissue: np.ndarray         # labels.TISSUE_CODE, 0 = background/CSF
    hemi: np.ndarray           # 1 left, 2 right
    ot: np.ndarray             # follow-up infarct (bool)
    maps: dict = field(default_factory=dict)   # name -> float32 volume
    info: dict = field(default_factory=dict)

    @property
    def voxel_ml(self):
        return float(abs(np.linalg.det(self.affine[:3, :3]))) / 1000.0

    @property
    def brain(self):
        """Tissue voxels (CSF and background excluded) inside the CTP field of view."""
        return (self.tissue > 0) & np.isfinite(self.maps.get("CBF", np.zeros(self.tissue.shape)))

    def add_map(self, name, path):
        self.maps[name] = _load(path)[1]

    def ctc(self):
        return _load(self.folder / "CTC_from_MR_4DPWI.nii.gz")[1]


def load_subject(root, num):
    d = Path(root) / f"training_{num}"
    img, seg = _load(d / "synthseg.nii.gz")
    seg = np.rint(seg).astype(np.int32)
    ot = _load(d / "OT.nii.gz")[1] > 0
    s = Subject(num=num, folder=d, affine=img.affine, seg=seg, tissue=tissue_map(seg),
                hemi=hemisphere_map(seg), ot=ot,
                info=json.loads((d / "prep_info.json").read_text(encoding="utf8")))
    for name, stem in VENDOR.items():
        p = d / f"{stem}.nii.gz"
        if p.exists():
            s.add_map(name, p)
    ncct = d / "ncct.nii.gz"
    if ncct.exists():
        s.add_map("NCCT", ncct)
    return s
