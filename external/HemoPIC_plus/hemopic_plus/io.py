"""NIfTI and path helpers (adapted from HemoPIC, MIT License, LDR Group @ JHU)."""
import json
import shutil
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

SEG4_REL = Path("hemopic_seg4") / "seg4_label_ref24.nii.gz"


def load_nifti(path):
    img = nib.load(str(path))
    return img, np.asanyarray(img.get_fdata())


def find_nii(base: Path, stem: str):
    for suffix in (".nii", ".nii.gz"):
        p = Path(base) / (stem + suffix)
        if p.exists():
            return p
    return None


def squeeze_ctc(ctc_raw):
    if ctc_raw.ndim == 5:
        if ctc_raw.shape[3] != 1:
            raise RuntimeError("CTC channel dim is not 1")
        return np.squeeze(ctc_raw, axis=3)
    if ctc_raw.ndim == 4:
        return ctc_raw
    raise RuntimeError("Unexpected CTC ndim")


def safe_dt_from_header(img):
    zooms = img.header.get_zooms()
    if zooms is None or len(zooms) < 4:
        return 1.0
    dt = float(zooms[3])
    if (not np.isfinite(dt)) or dt <= 0.0:
        return 1.0
    return dt


@lru_cache(maxsize=4)
def _load_tr_table(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    return {int(k): float(v) for k, v in json.loads(p.read_text(encoding="utf8")).items()}


def resolve_dt(cfg, ctc_img, patient_num: int):
    """Return (dt_seconds, source_label) for one subject."""
    if cfg.dt_source == "header":
        return safe_dt_from_header(ctc_img), "header"
    table = _load_tr_table(str(cfg.resolve("tr_table")))
    if int(patient_num) in table:
        return table[int(patient_num)], "dicom_tr"
    return float(cfg.tr_default), "tr_default"


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_nifti_like(ref_img, vol, out_path: Path, dtype=np.float32):
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(vol.astype(dtype), affine=ref_img.affine, header=hdr), str(out_path))


def subject_id(patient_num: int) -> str:
    return "training_" + str(int(patient_num))


def list_patients(root: Path):
    out = []
    for d in Path(root).glob("training_*"):
        try:
            out.append(int(d.name.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(out)
