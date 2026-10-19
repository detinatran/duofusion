"""[C6] Quality control of the automatic AIF: one panel per subject with the
selected AIF (before/after partial-volume rescaling), the VOF and HemoPIC's
tissue-mean arterial proxy, plus a CSV of AIF diagnostics.

python scripts/plot_aif.py configs/full_aif.yaml
python scripts/plot_aif.py configs/full_aif.yaml --patients 1 22 30
-> <out_root>/aif_qc/aif_qc.png, aif_qc.csv
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hemopic_plus.aif import select_aif  # noqa: E402
from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, list_patients, load_nifti, resolve_dt, squeeze_ctc, subject_id  # noqa: E402
from hemopic_plus.signal import choose_peak_threshold, compute_peak_map, smooth_curve_gaussian  # noqa: E402


def _work(args):
    cfg, pid = args
    base = cfg.resolve("data_root") / subject_id(pid)
    img, raw = load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))
    ctc = squeeze_ctc(raw)
    seg = load_nifti(base / SEG4_REL)[1] > 0
    dt, _ = resolve_dt(cfg, img, pid)
    t = np.arange(ctc.shape[3]) * dt
    show = seg & np.all(np.isfinite(ctc), axis=3)
    peak3 = compute_peak_map(ctc, cfg.baseline_frames)
    fit = show & (peak3 > choose_peak_threshold(peak3, show))
    tissue = smooth_curve_gaussian(ctc[fit].mean(axis=0), cfg.smooth_sigma_frames, cfg.baseline_frames, True)
    res = select_aif(ctc, show, t, cfg, seed=cfg.seed + pid)
    return pid, t, tissue, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--patients", type=int, nargs="*")
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    a = p.parse_args()
    ov = {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v is not None}
    cfg = load_config(a.config, ov)
    pids = a.patients or list_patients(cfg.resolve("data_root"))
    out = cfg.resolve("out_root") / "aif_qc"
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=cfg.workers or os.cpu_count() or 1) as ex:
        results = list(ex.map(_work, [(cfg, pid) for pid in pids]))

    cols = 6
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False)
    table = []
    for ax, (pid, t, tissue, res) in zip(axes.ravel(), results):
        ax.plot(t, tissue, color="0.5", lw=1.2, label="tissue mean (HemoPIC)")
        if res is not None:
            ax.plot(t, res["aif_unscaled"], color="tab:red", lw=0.8, ls="--", label="AIF raw")
            ax.plot(t, res["aif"], color="tab:red", lw=1.4, label="AIF (PVE corr.)")
            if res["vof"] is not None:
                ax.plot(t, res["vof"], color="tab:blue", lw=1.0, label="VOF")
            table.append({"patient": pid, "n_voxels": res["n_voxels"], "ttp_s": round(res["ttp_s"], 2),
                          "fwhm_s": round(res["fwhm_s"], 2), "first_moment_s": round(res["first_moment_s"], 2),
                          "tissue_first_moment_s": round(res["tissue_first_moment_s"], 2), "tissue_fwhm_s": round(res["tissue_fwhm_s"], 2),
                          "pve_scale": round(res["pve_scale"], 3),
                          "peak_ratio_aif_tissue": round(float(res["aif"].max() / max(tissue.max(), 1e-12)), 2)})
        else:
            table.append({"patient": pid, "n_voxels": 0})
        ax.set_title(f"training_{pid}" + ("" if res is not None else "  (no AIF -> tissue mean)"), fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(results):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "aif_qc.png", dpi=110)
    with open(out / "aif_qc.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=max((list(r.keys()) for r in table), key=len), restval="")
        w.writeheader()
        w.writerows(table)
    print(f"{sum(r['n_voxels'] > 0 for r in table)}/{len(results)} subjects with an AIF -> {out}")


if __name__ == "__main__":
    main()
