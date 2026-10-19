"""Run time per patient on one CPU thread: the regional HemoPIC+ fit, the voxel-level
fit of twin-vox, and oSVD with the same AIF for comparison.

python scripts/time_pipeline.py configs/full_aif.yaml --patients 4 22 35 [--out-root outputs_timing]

The regional fit is rerun from scratch into --out-root (the paper's outputs are
not touched); the voxel fit and oSVD use that fit, as scripts/eval_external.py.
Writes <out-root>/timing.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_external import voxel_fit  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, load_nifti, squeeze_ctc, subject_id  # noqa: E402
from hemopic_plus.osvd import osvd_maps  # noqa: E402
from hemopic_plus.signal import positive_excursion  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--patients", type=int, nargs="+", required=True)
    p.add_argument("--data-root")
    p.add_argument("--out-root", default="outputs_timing")
    a = p.parse_args()
    torch.set_num_threads(1)
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    data_root, out = cfg.resolve("data_root"), cfg.resolve("out_root")
    res = {"patients": {}}
    for pid in a.patients:
        sub = subject_id(pid)
        t0 = time.perf_counter()
        cmd = [sys.executable, "scripts/run_cohort.py", str(a.config), "--patients", str(pid), "--workers", "1", "--out-root", str(out)]
        if a.data_root:
            cmd += ["--data-root", a.data_root]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        t_roi = time.perf_counter() - t0
        fit_dir = out / cfg.name / sub
        base = data_root / sub
        seg = load_nifti(base / SEG4_REL)[1]
        ctc = squeeze_ctc(load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))[1])
        show = (seg > 0) & np.all(np.isfinite(ctc), axis=3)
        lab = nib.load(str(fit_dir / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
        t0 = time.perf_counter()
        voxel_fit(fit_dir, ctc, show, lab)
        t_vox = time.perf_counter() - t0
        info = json.loads((fit_dir / "run_info_core.json").read_text(encoding="utf8"))
        art = positive_excursion(np.load(fit_dir / "fit_curves.npz")["cart_norm"][None, :], cfg.baseline_frames)[0]
        t0 = time.perf_counter()
        curves = positive_excursion(ctc[show].astype(np.float64), cfg.baseline_frames)
        osvd_maps(curves, art, float(info["dt_seconds"]), k_h=cfg.k_h, rho=cfg.rho_brain)
        t_osvd = time.perf_counter() - t0
        res["patients"][sub] = {"voxels": int(show.sum()), "frames": int(ctc.shape[3]), "ROI fit s": t_roi,
                                "voxel fit s": t_vox, "oSVD s": t_osvd}
        print(sub, res["patients"][sub], flush=True)
    for k in ("ROI fit s", "voxel fit s", "oSVD s", "voxels"):
        res[f"median {k}"] = float(np.median([v[k] for v in res["patients"].values()]))
    res["threads"] = 1
    (out / "timing.json").write_text(json.dumps(res, indent=2), encoding="utf8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
