"""Central-volume maps from the voxel-level transport fit (run after run_voxel_pde.py).

python scripts/eval_voxel_maps.py configs/full_aif.yaml --workers 6

The voxel fit identifies the shape of each curve (its MTT) much better than its
area, and indicator dilution already gives the voxel CBV (full_aif_idvox_osvd_nc,
the [C8] map, AUC on par with oSVD). Fixed in advance, without looking at lesion
labels, twin-vox therefore combines
    CBV = indicator-dilution CBV, MTT = voxel MTT of the transport fit,
    CBF = 60 CBV / MTT   (central volume theorem),
for the compartment (vox_tank) and the neural PDE (vox_npde). Scored like
compare_osvd.py on the patients with oSVD maps, against oSVD (same AIF), the
hand-made map construction and the nested Optuna search.
Writes <out_root>/vox_pde/central_volume.{csv,json}.
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, load_nifti, subject_id  # noqa: E402

ID_CBV = "full_aif_idvox_osvd_nc"


def load(p):
    return nib.load(str(p)).get_fdata() if Path(p).exists() else None


def auc(vol, brain, ot, direction):
    if vol is None:
        return np.nan
    valid, score = _lesion_scores(vol, brain, direction)
    return roc_auc(ot[valid], score) if score is not None else np.nan


def work(args):
    pid, out, data_root, name = args
    sub = subject_id(pid)
    base = data_root / sub
    brain = load_nifti(base / SEG4_REL)[1] > 0
    ot = load_nifti(find_nii(base, "OT"))[1] > 0
    res = {"patient": pid}
    cbv_id = load(out / ID_CBV / sub / "HemoPIC_CBV.nii.gz")
    for var in ("tank", "npde"):
        mtt = load(out / f"{name}__vox_{var}" / sub / "VOX_MTT.nii.gz")
        if mtt is None or cbv_id is None:
            continue
        cbf = np.where(np.isfinite(mtt) & (mtt > 0), 60.0 * cbv_id / np.maximum(mtt, 1e-6), np.nan)
        for m, vol in (("CBF", cbf), ("CBV", cbv_id), ("MTT", mtt)):
            res[f"twin-vox {var} {m}"] = auc(vol, brain, ot, dict(MAPS)[m])
    for src, pat in (("oSVD", out / "osvd" / sub / "oSVD_{}.nii.gz"), ("hand-made", out / ID_CBV / sub / "HemoPIC_{}.nii.gz")):
        for m, direction in MAPS:
            res[f"{src} {m}"] = auc(load(str(pat).format(m)), brain, ot, direction)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out, data_root = cfg.resolve("out_root"), cfg.resolve("data_root")
    pids = sorted(int(d.name.split("_")[1]) for d in (out / "osvd").glob("training_*") if (d / "oSVD_CBF.nii.gz").exists())
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(work, [(pid, out, data_root, cfg.name) for pid in pids]))
    nested = {}
    for fd in sorted((out / "optuna_cv").glob("fold*")):
        for r in csv.DictReader(open(fd / "test_per_patient.csv", encoding="utf8")):
            if r["variant"].startswith("best1"):
                nested[int(r["patient"])] = {m: float(r[f"AUC_{m}"]) for m, _ in MAPS}
    for r in rows:
        for m, _ in MAPS:
            r[f"nested Optuna {m}"] = nested.get(r["patient"], {}).get(m, np.nan)
    methods = ["oSVD", "hand-made", "nested Optuna", "twin-vox tank", "twin-vox npde"]
    for r in rows:
        for meth in methods:
            vals = [r.get(f"{meth} {m}", np.nan) for m, _ in MAPS]
            r[f"{meth} mean"] = float(np.mean(vals)) if np.all(np.isfinite(vals)) else np.nan
    keys = [k for k in rows[0] if k != "patient"]
    od = out / "vox_pde"
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "central_volume.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["patient", *keys])
        for r in rows:
            w.writerow([r["patient"], *[r.get(k) for k in keys]])

    def paired(x, y):
        d = np.array([r.get(x, np.nan) - r.get(y, np.nan) for r in rows])
        d = d[np.isfinite(d)]
        return {"diff": float(d.mean()), "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, "n": int(d.size)}

    summ = {"n": len(rows), "mean": {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}, "paired": {}}
    for var in ("tank", "npde"):
        for ref in ("oSVD", "hand-made", "nested Optuna"):
            for m in ("CBF", "CBV", "MTT", "mean"):
                summ["paired"][f"twin-vox {var} - {ref} {m}"] = paired(f"twin-vox {var} {m}", f"{ref} {m}")
    (od / "central_volume.json").write_text(json.dumps(summ, indent=2), encoding="utf8")
    print(f"lesion ROC AUC on {len(rows)} patients with oSVD maps")
    for meth in methods:
        print(f"  {meth:18s} " + "  ".join(f"{m} {summ['mean'][f'{meth} {m}']:.3f}" for m in ("CBF", "CBV", "MTT", "mean")))
    for k, v in summ["paired"].items():
        print(f"  {k:44s} {v['diff']:+.4f}  p={v['p']:.2g}  n={v['n']}")


if __name__ == "__main__":
    main()
