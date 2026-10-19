"""Volume grids of the hybrid (sSVD core + twin relative-Tmax hypoperfusion) for every scan length.

python scripts/hybrid_grids.py <results_dir> <data_root> --yy 24 --lengths full 25 30 35 40 45 [--workers 8]

Reads ssvd<yy>_<full|T L>/training_X/SSVD_CBF.nii.gz and duofusion<yy>[_T L]/training_X/DF_Tmax_rel.nii.gz
and writes hybrid<yy>_<full|T L>/training_X.npz with core_grid (sSVD rCBF < tau over TAU_GRID,
reference = sSVD CBF median of the twin-healthy contralateral tissue), hypo_grid (twin
Tmax_rel > d over DT_GRID) and union_grid (tau x d), in the format of osvd_volumes.py.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from duofusion.data import load_subject  # noqa: E402
from duofusion.reference import affected_side  # noqa: E402

TAU_GRID = np.round(np.arange(0.10, 0.551, 0.05), 3)
DT_GRID = np.round(np.arange(0.0, 8.01, 0.5), 2)


def work(args):
    res, data_root, yy, suf, n = args
    sd = res / f"ssvd{yy}_{suf}" / f"training_{n}"
    dd = res / (f"duofusion{yy}" if suf == "full" else f"duofusion{yy}_{suf}") / f"training_{n}"
    od = res / f"hybrid{yy}_{suf}"
    if not ((sd / "SSVD_CBF.nii.gz").exists() and (dd / "DF_Tmax_rel.nii.gz").exists()):
        return 0
    s = load_subject(data_root, n)
    cbf = nib.load(str(sd / "SSVD_CBF.nii.gz")).get_fdata()
    trel = nib.load(str(dd / "DF_Tmax_rel.nii.gz")).get_fdata()
    valid = (s.tissue > 0) & np.isfinite(cbf) & (cbf > 0) & np.isfinite(trel)
    side = affected_side(s, valid & (trel > 4.0))
    contra = (s.hemi == 3 - side) if side else (s.hemi > 0)
    healthy = valid & (trel < 2.0)
    sel = healthy & contra if np.any(healthy & contra) else (healthy if np.any(healthy) else valid)
    r = cbf / float(np.median(cbf[sel]))
    cores = [valid & (r < t) for t in TAU_GRID]
    hypos = [valid & (trel > d) for d in DT_GRID]
    ml = s.voxel_ml
    np.savez_compressed(od / f"training_{n}.npz", tau_grid=TAU_GRID, dt_grid=DT_GRID,
                        core_grid=np.array([c.sum() for c in cores]) * ml,
                        hypo_grid=np.array([h.sum() for h in hypos]) * ml,
                        union_grid=np.array([[(c | h).sum() for h in hypos] for c in cores]) * ml)
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("data_root", type=Path)
    p.add_argument("--yy", default="24")
    p.add_argument("--lengths", nargs="+", default=["full"])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    for L in a.lengths:
        suf = "full" if L == "full" else f"T{L}"
        (a.results / f"hybrid{a.yy}_{suf}").mkdir(parents=True, exist_ok=True)
        src = a.results / f"ssvd{a.yy}_{suf}"
        nums = sorted(int(f.stem.split("_")[1]) for f in src.glob("training_*.npz"))
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            done = sum(ex.map(work, [(a.results, a.data_root, a.yy, suf, n) for n in nums]))
        print(f"hybrid{a.yy}_{suf}: {done} subjects", flush=True)


if __name__ == "__main__":
    main()
