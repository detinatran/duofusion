"""Copy of a prepared cohort with the CTP truncated to its first L seconds (a shorter scan).

python scripts/truncate_ctp.py <data_root> <out_root> --seconds 30

Every file is symlinked except CTC_from_MR_4DPWI.nii.gz, cut to the frames acquired in
the first L seconds; prep_info.json gets "truncated_s" and the new frame count. A scan
already shorter than L is kept whole (flag "truncated": false).
"""
import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=Path)
    p.add_argument("out_root", type=Path)
    p.add_argument("--seconds", type=float, required=True)
    a = p.parse_args()
    a.out_root.mkdir(parents=True, exist_ok=True)
    n = cut = 0
    for d in sorted(a.data_root.glob("training_*")):
        if not (d / "prep_info.json").exists():
            continue
        o = a.out_root / d.name
        if (o / "prep_info.json").exists():
            continue
        o.mkdir(exist_ok=True)
        for f in d.iterdir():
            if f.name in ("CTC_from_MR_4DPWI.nii.gz", "prep_info.json"):
                continue
            if not (o / f.name).exists():
                os.symlink(f.resolve(), o / f.name)
        img = nib.load(str(d / "CTC_from_MR_4DPWI.nii.gz"))
        dt = float(img.header.get_zooms()[3]) or 1.0
        keep = int(np.floor(a.seconds / dt)) + 1
        data = np.asarray(img.dataobj, np.float32)
        info = json.loads((d / "prep_info.json").read_text(encoding="utf8"))
        info["truncated"] = bool(data.shape[3] > keep)
        info["truncated_s"] = a.seconds
        if info["truncated"]:
            data = data[..., :keep]
            cut += 1
        info["frames"] = int(data.shape[3])
        nib.save(nib.Nifti1Image(data, img.affine, img.header), str(o / "CTC_from_MR_4DPWI.nii.gz"))
        (o / "prep_info.json").write_text(json.dumps(info, indent=1), encoding="utf8")
        n += 1
    print(f"{n} subjects written, {cut} truncated to {a.seconds:g} s")


if __name__ == "__main__":
    main()
