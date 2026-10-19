"""Figures for the voxel-level transport fit (run after run_voxel_pde.py and eval_voxel_maps.py).

python scripts/make_voxel_figures.py [--data-root ...] [--out-root outputs] [--patient 22]
Writes docs/figures/vox_maps_example.png (needs the maps of one patient) and
docs/figures/vox_auc.png (from docs/results/vox_pde/).
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import C, INK, INK2  # noqa: E402

from hemopic_plus.io import find_nii  # noqa: E402

FIG = ROOT / "docs" / "figures"
RES = ROOT / "docs" / "results" / "vox_pde"


def canon(p):
    return nib.as_closest_canonical(nib.load(str(p))).get_fdata()


def fig_maps(out, data_root, pid):
    base = data_root / f"training_{pid}"
    sub = f"training_{pid}"
    seg = canon(base / "hemopic_seg4" / "seg4_label_ref24.nii.gz") > 0
    ot = canon(base / "OT.nii") > 0
    z = int(np.argmax(ot.sum(axis=(0, 1))))
    xs, ys = np.where(seg[:, :, z])
    crop = (slice(xs.min() - 3, xs.max() + 4), slice(ys.min() - 3, ys.max() + 4))
    ctc = canon(find_nii(base, "CTC_from_MR_4DPWI"))
    sl = lambda v: v[:, :, z][crop].T  # noqa: E731
    brain = sl(seg)

    def rel(v):
        v = sl(v)
        ok = brain & np.isfinite(v) & (v > 0)
        return np.where(ok, v / np.median(v[ok]), np.nan)

    def rel_s(v):
        v = sl(v)
        ok = brain & np.isfinite(v)
        return np.where(ok, v - np.median(v[ok]), np.nan)

    curves = np.where(np.isfinite(ctc), ctc, 0.0)
    ttp = np.argmax(gaussian_filter1d(curves, 1.0, axis=3), axis=3).astype(float)
    cbv_id = canon(out / "full_aif_idvox_osvd_nc" / sub / "HemoPIC_CBV.nii.gz")
    mtt_t = canon(out / "full_aif__vox_tank" / sub / "VOX_MTT.nii.gz")
    mtt_n = canon(out / "full_aif__vox_npde" / sub / "VOX_MTT.nii.gz")
    cbf_v = np.where(mtt_t > 0, 60.0 * cbv_id / np.maximum(mtt_t, 1e-6), np.nan)
    z_dt = np.load(out / "full_aif" / sub / "fit_curves.npz")
    dt = float(z_dt["dt"])
    panels = [
        [("oSVD CBF", rel(canon(out / "osvd" / sub / "oSVD_CBF.nii.gz")), "Blues", 0, 2.5),
         ("oSVD MTT", rel(canon(out / "osvd" / sub / "oSVD_MTT.nii.gz")), "Oranges", 0, 2.5),
         ("ROI x texture MTT (hand-made)", rel(canon(out / "full_aif_idvox_osvd_nc" / sub / "HemoPIC_MTT.nii.gz")), "Oranges", 0, 2.5),
         ("voxel TTP (s, vs brain median)", rel_s(ttp * dt), "Purples", -2, 10)],
        [("twin-vox CBF = 60 CBV / MTT", rel(cbf_v), "Blues", 0, 2.5),
         ("voxel MTT, compartment", rel(mtt_t), "Oranges", 0, 2.5),
         ("voxel MTT, neural PDE", rel(mtt_n), "Oranges", 0, 2.5),
         ("voxel delay (s, vs brain median)", rel_s(canon(out / "full_aif__vox_tank" / sub / "VOX_delay.nii.gz")), "Purples", -2, 10)],
    ]
    o2 = sl(ot.astype(float))
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.2))
    for i, row in enumerate(panels):
        for j, (title, img, cmap, vmin, vmax) in enumerate(row):
            ax = axes[i, j]
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("#fcfcfb")
            im = ax.imshow(img, origin="lower", cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.contour(o2, levels=[0.5], colors=[INK], linewidths=0.8, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(title, fontsize=8.5, color=INK)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    axes[0, 0].set_ylabel("references", fontsize=9, color=INK)
    axes[1, 0].set_ylabel("voxel-level fit", fontsize=9, color=INK)
    fig.suptitle(f"training_{pid}, slice {z} (black contour = final infarct); CBF and MTT relative to the brain median",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "vox_maps_example.png", dpi=150)
    plt.close(fig)


def fig_auc():
    rows = list(csv.DictReader(open(RES / "central_volume.csv", encoding="utf8")))
    vox = {int(r["patient"]): r for r in csv.DictReader(open(RES / "per_patient.csv", encoding="utf8"))}
    methods = [("oSVD", "oSVD (same AIF)"), ("hand-made", "ROI x texture, hand-made"), ("nested Optuna", "ROI x texture, nested Optuna"),
               ("twin-vox npde", "twin-vox, neural PDE"), ("twin-vox tank", "twin-vox, compartment")]
    maps = ["CBF", "CBV", "MTT", "mean"]
    rng = np.random.default_rng(0)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 3.6), gridspec_kw={"width_ratios": [3.2, 1.3]})
    wbar = 0.16
    for k, (key, label) in enumerate(methods):
        for j, m in enumerate(maps):
            v = np.array([float(r[f"{key} {m}"]) for r in rows if r.get(f"{key} {m}") not in ("", "nan", None)])
            v = v[np.isfinite(v)]
            boots = [rng.choice(v, v.size).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            x = j + (k - 2) * wbar
            ax.bar(x, v.mean(), wbar * 0.9, color=["#b8b6b0", "#8fa8c8", C[0], C[2], C[1]][k], label=label if j == 0 else None)
            ax.plot([x, x], [lo, hi], color=INK2, lw=0.8)
    ax.set_xticks(range(len(maps)))
    ax.set_xticklabels(["CBF (low)", "CBV (low)", "MTT (high)", "mean of the three"])
    ax.set_ylim(0.55, 0.93)
    ax.set_ylabel("lesion ROC AUC (36 patients)")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.set_title("Maps (bars: mean, lines: 95% bootstrap CI)", loc="left", fontsize=9, color=INK)
    timing = [("oSVD Tmax", "oSVD Tmax"), ("vox_npde delay", "voxel delay, neural PDE"),
              ("vox_tank delay", "voxel delay, compartment"), ("TTP", "voxel TTP")]
    pids = [int(r["patient"]) for r in rows]
    for k, (key, label) in enumerate(timing):
        v = np.array([float(vox[p][key]) for p in pids if p in vox and vox[p][key] not in ("", "nan")])
        v = v[np.isfinite(v)]
        boots = [rng.choice(v, v.size).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ax2.barh(k, v.mean(), 0.6, color=["#b8b6b0", C[2], C[1], C[3]][k])
        ax2.plot([lo, hi], [k, k], color=INK2, lw=0.8)
    ax2.set_yticks(range(len(timing)))
    ax2.set_yticklabels([lbl for _, lbl in timing], fontsize=8)
    ax2.set_xlim(0.55, 0.85)
    ax2.set_xlabel("lesion ROC AUC (36 patients)")
    ax2.set_title("Timing maps (late = lesion)", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "vox_auc.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=str(ROOT / "outputs"))
    p.add_argument("--data-root", default=str(ROOT.parent / "HemoPIC" / "data" / "ISLES2017_Training"))
    p.add_argument("--patient", type=int, default=22)
    a = p.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    out, data_root = Path(a.out_root), Path(a.data_root)
    if (out / "full_aif__vox_tank" / f"training_{a.patient}" / "VOX_MTT.nii.gz").exists():
        fig_maps(out, data_root, a.patient)
    fig_auc()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
