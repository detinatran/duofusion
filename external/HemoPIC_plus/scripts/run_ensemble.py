"""Pipeline uncertainty: rerun the same experiment with other k-means seeds and
other numbers of clusters, and measure how much the maps move.

python scripts/run_ensemble.py configs/full.yaml --data-root ../HemoPIC/data/ISLES2017_Training

Members: seeds 0..4 at the config's (K_GM, K_WM), plus (K_GM, K_WM) = (6, 4) and
(10, 8) at seed 0. Member "s0" is the experiment itself.
Writes <out_root>/<name>__ensemble/<member>/training_XX/ (fits),
       <out_root>/<name>/training_XX/ens/HemoPIC_{CBF,CBV,MTT}_ens_sd.nii.gz,
       <out_root>/<name>/eval_uq/ensemble.json (summary, incl. lesion AUC per member)
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, subject_id  # noqa: E402
from hemopic_plus.pipeline import run_subject  # noqa: E402


def members(cfg):
    out = [(f"s{s}", s, cfg.k_gm, cfg.k_wm) for s in range(5)]
    out += [("k6_4", 0, 6, 4), ("k10_8", 0, 10, 8)]
    return out


def _fit(args):
    cfg, pid, out_root = args
    try:
        run_subject(cfg, pid, out_root)
        return pid, None
    except Exception as e:
        return pid, repr(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--workers", type=int)
    p.add_argument("--patients", type=int, nargs="*")
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    base = cfg.resolve("out_root") / cfg.name
    ens_root = cfg.resolve("out_root") / f"{cfg.name}__ensemble"
    pids = a.patients or list_patients(base)
    mem = members(cfg)

    jobs = []
    for name, seed, kg, kw in mem:
        if name == "s0":
            continue
        mcfg = replace(cfg, seed=seed, k_gm=kg, k_wm=kw)
        for pid in pids:
            if not (ens_root / name / subject_id(pid) / "run_info_core.json").exists():
                jobs.append((mcfg, pid, ens_root / name))
    print(f"[{cfg.name}] ensemble: {len(mem)} members, {len(jobs)} fits to run")
    with ProcessPoolExecutor(max_workers=a.workers or os.cpu_count() or 1) as ex:
        for pid, err in ex.map(_fit, jobs):
            if err:
                print("  FAIL", pid, err)

    data_root = cfg.resolve("data_root")
    auc = {name: {m: [] for m, _ in MAPS} for name, *_ in mem}
    rel_sd = {m: [] for m, _ in MAPS}
    rel_post = {m: [] for m, _ in MAPS}
    for pid in pids:
        sub = subject_id(pid)
        dirs = {name: (base / sub if name == "s0" else ens_root / name / sub) for name, *_ in mem}
        seg = nib.load(str(data_root / sub / SEG4_REL)).get_fdata() > 0
        ot = nib.load(str(find_nii(data_root / sub, "OT"))).get_fdata() > 0
        (base / sub / "ens").mkdir(exist_ok=True)
        ref = nib.load(str(base / sub / "HemoPIC_CBF.nii.gz"))
        for m, direction in MAPS:
            vols = {n: nib.load(str(d / f"HemoPIC_{m}.nii.gz")).get_fdata() for n, d in dirs.items()}
            for n, v in vols.items():
                valid, score = _lesion_scores(v, seg, direction)
                auc[n][m].append(roc_auc(ot[valid], score) if score is not None else np.nan)
            stack = np.stack(list(vols.values()))
            mean = np.nanmean(stack, axis=0)
            sd = np.nanstd(stack, axis=0)
            nib.save(nib.Nifti1Image(sd.astype(np.float32), ref.affine), str(base / sub / "ens" / f"HemoPIC_{m}_ens_sd.nii.gz"))
            ok = seg & np.isfinite(mean) & (mean > 0)
            rel_sd[m].append(float(np.nanmedian(sd[ok] / mean[ok])))
            post = base / sub / "uq" / f"HemoPIC_{m}_sd.nii.gz"
            if post.exists():
                ps = nib.load(str(post)).get_fdata()
                v0 = vols["s0"]
                okp = ok & np.isfinite(ps) & (v0 > 0)
                rel_post[m].append(float(np.nanmedian(ps[okp] / v0[okp])))
    summary = {
        "members": [n for n, *_ in mem],
        "auc_mean": {n: {m: float(np.nanmean(v)) for m, v in d.items()} for n, d in auc.items()},
        "auc_range_over_members": {m: [float(min(np.nanmean(auc[n][m]) for n in auc)), float(max(np.nanmean(auc[n][m]) for n in auc))]
                                   for m, _ in MAPS},
        "median_rel_sd_pipeline": {m: float(np.median(v)) for m, v in rel_sd.items()},
        "median_rel_sd_posterior": {m: (float(np.median(v)) if v else None) for m, v in rel_post.items()},
    }
    (base / "eval_uq").mkdir(exist_ok=True)
    (base / "eval_uq" / "ensemble.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(f"[{cfg.name}] lesion AUC per member (CBF / CBV / MTT):")
    for n in summary["members"]:
        d = summary["auc_mean"][n]
        print(f"  {n:6s} {d['CBF']:.4f} / {d['CBV']:.4f} / {d['MTT']:.4f}")
    print("  voxel relative sd, pipeline vs posterior: " + "  ".join(
        f"{m} {summary['median_rel_sd_pipeline'][m]:.1%} vs "
        f"{'–' if summary['median_rel_sd_posterior'][m] is None else format(summary['median_rel_sd_posterior'][m], '.1%')}"
        for m, _ in MAPS))


if __name__ == "__main__":
    main()
