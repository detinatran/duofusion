"""Independent oSVD reference with the [C6] AIF, to locate the CVT gap.

python scripts/osvd_check.py configs/full_aif.yaml --compare baseline full full_aif

For every subject with an accepted AIF: oSVD CBF/CBV/MTT maps
(<out_root>/osvd/training_XX/oSVD_*.nii.gz), regional mean MTT (GM, WM,
lesion, brain) for ISLES Deconv, oSVD and each HemoPIC(+) experiment, and lesion
ROC AUC of the oSVD maps. Writes <out_root>/osvd/{regional_mtt.csv,roc.csv,summary.md}.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from hemopic_plus.aif import select_aif  # noqa: E402
from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.evaluation import MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, load_nifti, resolve_dt, save_nifti_like, squeeze_ctc, subject_id  # noqa: E402
from hemopic_plus.osvd import osvd_maps  # noqa: E402
from hemopic_plus.signal import positive_excursion  # noqa: E402

REGIONS = ("GM", "WM", "Lesion", "Brain")


def _regional(vol, masks):
    out = {}
    for reg, m in masks.items():
        v = vol[m]
        v = v[np.isfinite(v) & (v > 0)]
        out[reg] = float(np.mean(v)) if v.size >= 20 else float("nan")
    return out


def _work(args):
    cfg, pid, out_root, compare = args
    base = cfg.resolve("data_root") / subject_id(pid)
    img, raw = load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))
    ctc = squeeze_ctc(raw)
    seg = load_nifti(base / SEG4_REL)[1].astype(int)
    dt, _ = resolve_dt(cfg, img, pid)
    t = np.arange(ctc.shape[3]) * dt
    show = (seg > 0) & np.all(np.isfinite(ctc), axis=3)
    res = select_aif(ctc, show, t, cfg, seed=int(cfg.seed) + pid)
    if res is None:
        return pid, None, None
    aif = positive_excursion(res["aif"][None, :], cfg.baseline_frames)[0]
    curves = positive_excursion(ctc[show].astype(np.float64), cfg.baseline_frames)
    cbf, cbv, mtt = osvd_maps(curves, aif, dt, k_h=cfg.k_h, rho=cfg.rho_brain)

    ref_img, _ = load_nifti(find_nii(base, "MR_rCBF"))
    odir = out_root / "osvd" / subject_id(pid)
    odir.mkdir(parents=True, exist_ok=True)
    vols = {}
    for tag, v in (("CBF", cbf), ("CBV", cbv), ("MTT", mtt)):
        vol = np.full(seg.shape, np.nan)
        vol[show] = v
        vols[tag] = vol
        save_nifti_like(ref_img, vol, odir / f"oSVD_{tag}.nii.gz")

    ot_path = find_nii(base, "OT")
    ot = load_nifti(ot_path)[1] > 0
    masks = {"GM": seg == 1, "WM": seg == 2, "Lesion": ot & (seg > 0), "Brain": seg > 0}
    rows = [{"patient": pid, "method": "isles", **_regional(load_nifti(find_nii(base, "MR_MTT"))[1], masks)},
            {"patient": pid, "method": "osvd", **_regional(vols["MTT"], masks)}]
    for exp in compare:
        p = out_root / exp / subject_id(pid) / "HemoPIC_MTT.nii.gz"
        if p.exists():
            rows.append({"patient": pid, "method": exp, **_regional(load_nifti(p)[1], masks)})

    brain = seg > 0
    roc = []
    for name, direction in MAPS:
        valid, score = _lesion_scores(vols[name], brain, direction)
        roc.append({"patient": pid, "map": name, "auc": roc_auc(ot[valid], score) if score is not None else float("nan")})
    return pid, rows, roc


def _fit(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return float("nan"), float("nan"), int(ok.sum())
    return float(np.polyfit(x[ok], y[ok], 1)[0]), float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--compare", nargs="*", default=["baseline", "full", "full_aif"])
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    a = p.parse_args()
    ov = {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v is not None}
    cfg = load_config(a.config, ov)
    out_root = cfg.resolve("out_root")
    pids = a.patients or list_patients(cfg.resolve("data_root"))
    with ProcessPoolExecutor(max_workers=cfg.workers or os.cpu_count() or 1) as ex:
        results = list(ex.map(_work, [(cfg, pid, out_root, a.compare) for pid in pids]))
    rows = [r for _, rs, _ in results if rs for r in rs]
    rocs = [r for _, _, rc in results if rc for r in rc]
    used = sorted(pid for pid, rs, _ in results if rs)
    od = out_root / "osvd"
    with open(od / "regional_mtt.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "method", *REGIONS])
        w.writeheader()
        w.writerows(rows)
    with open(od / "roc.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "map", "auc"])
        w.writeheader()
        w.writerows(rocs)

    def series(method, reg):
        d = {r["patient"]: r[reg] for r in rows if r["method"] == method}
        return np.array([d.get(pid, np.nan) for pid in used])

    pairs = [("isles", "osvd")] + [(x, e) for e in a.compare for x in ("osvd", "isles")]
    lines = [f"oSVD with the C6 AIF on {len(used)} subjects (subjects without an accepted AIF are skipped).", "",
             "Patient-level regional mean MTT, y vs x: slope / Pearson r (n)", "",
             "| x | y | " + " | ".join(REGIONS) + " | mean(y)/mean(x) Brain |", "|---|---|" + "---|" * len(REGIONS) + "---|"]
    for x, y in pairs:
        cells = []
        for reg in REGIONS:
            s, r, n = _fit(series(x, reg), series(y, reg))
            cells.append(f"{s:.2f} / {r:.2f} ({n})")
        bx, by = series(x, "Brain"), series(y, "Brain")
        ok = np.isfinite(bx) & np.isfinite(by)
        lines.append(f"| {x} | {y} | " + " | ".join(cells) + f" | {by[ok].mean() / bx[ok].mean():.2f} |")
    lines += ["", "Lesion ROC AUC of oSVD maps (relative to brain median), mean over subjects:", ""]
    for name, _ in MAPS:
        v = np.array([r["auc"] for r in rocs if r["map"] == name])
        lines.append(f"- {name}: {np.nanmean(v):.4f} (n={int(np.isfinite(v).sum())})")
    text = "\n".join(lines)
    (od / "summary.md").write_text(text + "\n", encoding="utf8")
    print(text)


if __name__ == "__main__":
    main()
