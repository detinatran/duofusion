"""oSVD settings for CT perfusion on the external set (run after eval_external.py).

python scripts/osvd_grid_ct.py --data-root <ISLES2024_Training> [--prefix isles24] [--workers 20]

oSVD was set up for the MR set (oscillation index 0.095, no smoothing). On CT
perfusion it is run with the HemoPIC+ AIF under several settings: other
oscillation-index targets, block-circulant SVD with a fixed truncation (sSVD,
the usual choice for CT), temporal smoothing and in-plane spatial smoothing
within the brain (as scripts/smooth_external.py). Lesion ROC AUC as
eval_external.py. The best setting on the mean AUC is compared, patient by
patient, with the methods of <out_root>/<prefix>_external/per_patient.csv.
Choosing it on the test labels favours oSVD.
Writes <out_root>/<prefix>_external/osvd_grid.{json,csv} and osvd_grid_summary.json.
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

import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter, gaussian_filter1d  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, load_nifti, squeeze_ctc, subject_id  # noqa: E402
from hemopic_plus.osvd import osvd_maps  # noqa: E402
from hemopic_plus.signal import positive_excursion  # noqa: E402

SETTINGS = {
    "OI 0.095 (MR setting)": {},
    "OI 0.035": {"osc": 0.035},
    "OI 0.2": {"osc": 0.2},
    "sSVD 10%": {"ssvd": 0.1},
    "sSVD 20%": {"ssvd": 0.2},
    "time smoothing 1 frame": {"st": 1.0},
    "space smoothing 1.5 voxels": {"ss": 1.5},
    "space smoothing 3 voxels": {"ss": 3.0},
    "space 3 voxels + sSVD 20%": {"ss": 3.0, "ssvd": 0.2},
    "space 3 voxels + time 1 frame": {"ss": 3.0, "st": 1.0},
}
METHODS = ("vendor", "HemoPIC", "HemoPIC+", "HemoPIC+ ID-CBV oSVD texture", "twin-vox")


def auc(vol, brain, ot, direction):
    valid, score = _lesion_scores(vol, brain, direction)
    return roc_auc(ot[valid], score) if score is not None else np.nan


def work(args):
    pid, a = args
    cfg = load_config(Path("configs") / f"{a.prefix}_full_aif.yaml", {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out, sub = cfg.resolve("out_root"), subject_id(pid)
    base, fit_dir = cfg.resolve("data_root") / sub, out / f"{a.prefix}_full_aif" / sub
    if not (fit_dir / "run_info_core.json").exists():
        return None
    info = json.loads((fit_dir / "run_info_core.json").read_text(encoding="utf8"))
    seg = load_nifti(base / SEG4_REL)[1]
    brain = seg > 0
    ot = load_nifti(find_nii(base, "OT"))[1] > 0
    if not (ot & brain).any():
        return None
    ctc = squeeze_ctc(load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))[1]).astype(np.float64)
    show = brain & np.all(np.isfinite(ctc), axis=3)
    art = positive_excursion(np.load(fit_dir / "fit_curves.npz")["cart_norm"][None, :], cfg.baseline_frames)[0]
    res = {"patient": pid}
    for name, s in SETTINGS.items():
        x = ctc
        if s.get("ss"):
            num = gaussian_filter(np.where(show[..., None], x, 0.0), sigma=(s["ss"], s["ss"], 0, 0))
            den = gaussian_filter(show.astype(np.float64), sigma=(s["ss"], s["ss"], 0))[..., None]
            x = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
        cur = x[show]
        if s.get("st"):
            cur = gaussian_filter1d(cur, s["st"], axis=1, mode="nearest")
        cur = positive_excursion(cur, cfg.baseline_frames)
        kw = {"k_h": cfg.k_h, "rho": cfg.rho_brain}
        if s.get("ssvd"):
            kw.update(thresholds=(s["ssvd"],), osc_target=np.inf)
        elif s.get("osc"):
            kw.update(osc_target=s["osc"])
        o = dict(zip(("CBF", "CBV", "MTT"), osvd_maps(cur, art, float(info["dt_seconds"]), **kw)))
        for m, direction in MAPS:
            v = np.full(seg.shape, np.nan)
            v[show] = o[m]
            res[f"{name} {m}"] = auc(v, brain, ot, direction)
        res[f"{name} mean"] = float(np.mean([res[f"{name} {m}"] for m, _ in MAPS]))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-root")
    p.add_argument("--prefix", default="isles24")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--compare", type=Path, help="per_patient.csv of eval_external.py to compare with (default: this run's)")
    a = p.parse_args()
    cfg = load_config(Path("configs") / f"{a.prefix}_full_aif.yaml", {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    od = cfg.resolve("out_root") / f"{a.prefix}_external"
    od.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, [(pid, a) for pid in sorted(list_patients(cfg.resolve("data_root")))]):
            if r:
                rows.append(r)
    (od / "osvd_grid.json").write_text(json.dumps(rows), encoding="utf8")
    keys = [k for k in rows[0] if k != "patient"]
    with open(od / "osvd_grid.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.writer(f)
        wr.writerow(["patient", *keys])
        for r in rows:
            wr.writerow([r["patient"], *[r[k] for k in keys]])
    summ = {"n": len(rows), "settings": {n: {m: float(np.nanmean([r[f"{n} {m}"] for r in rows])) for m in ("CBF", "CBV", "MTT", "mean")}
                                         for n in SETTINGS}}
    best = max(SETTINGS, key=lambda n: summ["settings"][n]["mean"])
    summ["best"] = best
    comp = a.compare or od / "per_patient.csv"
    per = {int(r["patient"]): r for r in csv.DictReader(open(comp, encoding="utf8"))}
    tests = {}
    for meth in METHODS:
        for m in ("CBF", "CBV", "MTT", "mean"):
            d = np.array([float(per[r["patient"]][f"{meth} {m}"]) - r[f"{best} {m}"] for r in rows if r["patient"] in per])
            d = d[np.isfinite(d)]
            if d.size > 5:
                tests[f"{meth} - oSVD best {m}"] = {"diff": float(d.mean()), "p": float(wilcoxon(d).pvalue), "n_better": int((d > 0).sum()), "n": int(d.size)}
    summ["compared_with"] = str(comp)
    summ["paired"] = tests
    (od / "osvd_grid_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
