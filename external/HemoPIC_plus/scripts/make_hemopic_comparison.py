"""HemoPIC versus twin-vox: maps of one patient and per-patient lesion AUC
(run after run_voxel_pde.py and eval_voxel_maps.py).

python scripts/make_hemopic_comparison.py [--data-root ...] [--out-root outputs] [--patient 13]

CBF, CBV and MTT relative to the brain median, in rows: the reference maps
distributed with ISLES 2017 (MR_rCBF, MR_rCBV, MR_MTT), HemoPIC (the original
pipeline, outputs/baseline) and twin-vox (indicator-dilution CBV, voxel MTT,
CBF = 60 CBV / MTT), with the final infarct and each map's lesion AUC for this
patient (lower left corner, the best of each column in bold; --maps-width,
3.1 in by default), on the slice
with the largest infarct area. Second figure: lesion AUC of
every patient with oSVD maps, HemoPIC on x and twin-vox on y, one panel per
map (above the diagonal: twin-vox better; one column, 3.35 in). Scored like
compare_osvd.py.
Writes docs/figures/hemopic_vs_twin_{maps,auc}.png and docs/results/vox_pde/hemopic_vs_twin.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import C, INK, INK2  # noqa: E402

from hemopic_plus.evaluation import MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, load_nifti  # noqa: E402

FIG = ROOT / "docs" / "figures"
RES = ROOT / "docs" / "results" / "vox_pde"
ISLES = {"CBF": "MR_rCBF", "CBV": "MR_rCBV", "MTT": "MR_MTT"}
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6})


def twin_vox(out, sub):
    cbv = nib.load(str(out / "full_aif_idvox_osvd_nc" / sub / "HemoPIC_CBV.nii.gz")).get_fdata()
    mtt = nib.load(str(out / "full_aif__vox_tank" / sub / "VOX_MTT.nii.gz")).get_fdata()
    cbf = np.where(np.isfinite(mtt) & (mtt > 0), 60.0 * cbv / np.maximum(mtt, 1e-6), np.nan)
    return {"CBF": cbf, "CBV": cbv, "MTT": mtt}


def hemopic(out, sub):
    return {m: nib.load(str(out / "baseline" / sub / f"HemoPIC_{m}.nii.gz")).get_fdata() for m, _ in MAPS}


def isles(base):
    return {m: load_nifti(find_nii(base, f))[1].astype(float) for m, f in ISLES.items()}


def aucs(vols, brain, ot):
    res = {}
    for m, direction in MAPS:
        valid, score = _lesion_scores(vols[m], brain, direction)
        res[m] = roc_auc(ot[valid], score) if score is not None else np.nan
    return res


def fig_maps(out, data_root, pid, auc_vox, auc_hem, path, width=3.35):
    sub = f"training_{pid}"
    base = data_root / sub
    canon = lambda v, ref: nib.as_closest_canonical(nib.Nifti1Image(v, nib.load(str(ref)).affine)).get_fdata()  # noqa: E731
    ref = base / "hemopic_seg4" / "seg4_label_ref24.nii.gz"
    seg = canon(nib.load(str(ref)).get_fdata(), ref) > 0
    ot = canon(load_nifti(find_nii(base, "OT"))[1].astype(float), ref) > 0
    z = int(np.argmax(ot.sum(axis=(0, 1))))
    xs, ys = np.where(seg[:, :, z])
    crop = (slice(xs.min() - 3, xs.max() + 4), slice(ys.min() - 3, ys.max() + 4))
    brain_sl = seg[:, :, z][crop].T
    o2 = ot[:, :, z][crop].T.astype(float)
    brain = load_nifti(base / SEG4_REL)[1] > 0
    ref_vols = isles(base)
    rows = [("ISLES ref.", ref_vols, aucs(ref_vols, brain, load_nifti(find_nii(base, "OT"))[1] > 0)),
            ("HemoPIC [4]", hemopic(out, sub), auc_hem), ("HemoPIC+", twin_vox(out, sub), auc_vox)]
    best = {m: max(r[2][m] for r in rows) for m, _ in MAPS}
    W, lab_w, gap, cb_gap, cb_w, cb_lab = width, 0.17, 0.03, 0.06, 0.07, 0.3
    pw = (W - lab_w - 2 * gap - cb_gap - cb_w - cb_lab) / 3
    ph = pw * brain_sl.shape[0] / brain_sl.shape[1]
    auc_h, title_h, gap_y = 0.0, 0.12, 0.025
    H = title_h + 3 * (ph + auc_h) + 2 * gap_y + 0.01
    fig = plt.figure(figsize=(W, H))
    cm = plt.get_cmap("Blues").copy()
    cm.set_bad("white")
    im = None
    for i, (rname, vols, auc) in enumerate(rows):
        y0 = H - title_h - (i + 1) * ph - i * (auc_h + gap_y)
        for j, (m, _) in enumerate(MAPS):
            ax = fig.add_axes([(lab_w + j * (pw + gap)) / W, y0 / H, pw / W, ph / H])
            v = canon(vols[m], ref)[:, :, z][crop].T
            ok = brain_sl & np.isfinite(v) & (v > 0)
            img = np.where(ok, v / np.median(v[ok]), np.nan)
            im = ax.imshow(img, origin="lower", cmap=cm, vmin=0, vmax=2.5, interpolation="nearest", aspect="auto")
            ax.contour(o2, levels=[0.5], colors=["#c8372d"], linewidths=0.6, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(m, pad=2, color=INK)
            top = auc[m] >= best[m] - 1e-9
            ax.text(0.0, 0.0, f"{auc[m]:.2f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=6,
                    color=INK if top else INK2, fontweight="bold" if top else "normal")
        fig.text(0.5 * lab_w / W, (y0 + 0.5 * ph) / H, rname, rotation=90, ha="center", va="center",
                 fontsize=7, fontweight="bold", color=INK)
    y_lo, y_hi = H - title_h - 3 * ph - 2 * (auc_h + gap_y), H - title_h
    cax = fig.add_axes([(W - cb_lab - cb_w) / W, (y_lo + 0.2 * (y_hi - y_lo)) / H, cb_w / W, 0.6 * (y_hi - y_lo) / H])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 1, 2])
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_label("/ brain median", fontsize=5.5, labelpad=2)
    fig.savefig(path, dpi=400)
    plt.close(fig)


def fig_maps_wide(out, data_root, pid, auc_vox, auc_hem, path, width=7.0, fs=7):
    """The panels of fig_maps in one row for a two-column-wide figure: three groups
    (ISLES ref., HemoPIC, twin-vox) of CBF, CBV and MTT, lesion AUC under each map."""
    sub = f"training_{pid}"
    base = data_root / sub
    canon = lambda v, ref: nib.as_closest_canonical(nib.Nifti1Image(v, nib.load(str(ref)).affine)).get_fdata()  # noqa: E731
    ref = base / "hemopic_seg4" / "seg4_label_ref24.nii.gz"
    seg = canon(nib.load(str(ref)).get_fdata(), ref) > 0
    ot = canon(load_nifti(find_nii(base, "OT"))[1].astype(float), ref) > 0
    z = int(np.argmax(ot.sum(axis=(0, 1))))
    xs, ys = np.where(seg[:, :, z])
    crop = (slice(xs.min() - 3, xs.max() + 4), slice(ys.min() - 3, ys.max() + 4))
    brain_sl = seg[:, :, z][crop].T
    o2 = ot[:, :, z][crop].T.astype(float)
    brain = load_nifti(base / SEG4_REL)[1] > 0
    ref_vols = isles(base)
    groups = [("ISLES maps", ref_vols, aucs(ref_vols, brain, load_nifti(find_nii(base, "OT"))[1] > 0)),
              ("HemoPIC", hemopic(out, sub), auc_hem), ("HemoPIC+", twin_vox(out, sub), auc_vox)]
    best = {m: max(g[2][m] for g in groups) for m, _ in MAPS}
    gap, ggap, cb_gap, cb_w, cb_lab = 0.03, 0.14, 0.08, 0.08, 0.42
    pw = (width - 6 * gap - 2 * ggap - cb_gap - cb_w - cb_lab) / 9
    ph = pw * brain_sl.shape[0] / brain_sl.shape[1]
    gt_h, mt_h, auc_h = 0.17, 0.15, 0.15
    H = gt_h + mt_h + ph + auc_h + 0.02
    fig = plt.figure(figsize=(width, H))
    cm = plt.get_cmap("Blues").copy()
    cm.set_bad("white")
    im = None
    y0 = auc_h
    for g, (gname, vols, auc) in enumerate(groups):
        x_g = g * (3 * pw + 2 * gap + ggap)
        fig.text((x_g + 1.5 * pw + gap) / width, (H - 0.5 * gt_h) / H, gname, ha="center", va="center",
                 fontsize=fs + 0.5, fontweight="bold", color=INK)
        for j, (m, _) in enumerate(MAPS):
            ax = fig.add_axes([(x_g + j * (pw + gap)) / width, y0 / H, pw / width, ph / H])
            v = canon(vols[m], ref)[:, :, z][crop].T
            ok = brain_sl & np.isfinite(v) & (v > 0)
            img = np.where(ok, v / np.median(v[ok]), np.nan)
            im = ax.imshow(img, origin="lower", cmap=cm, vmin=0, vmax=2.5, interpolation="nearest", aspect="auto")
            ax.contour(o2, levels=[0.5], colors=["#c8372d"], linewidths=0.7, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(m, pad=2, fontsize=fs, color=INK)
            top = auc[m] >= best[m] - 1e-9
            fig.text((x_g + j * (pw + gap) + 0.5 * pw) / width, 0.5 * auc_h / H, f"AUC {auc[m]:.2f}", ha="center",
                     va="center", fontsize=fs, color=INK if top else INK2, fontweight="bold" if top else "normal")
    cax = fig.add_axes([(width - cb_lab - cb_w) / width, (y0 + 0.1 * ph) / H, cb_w / width, 0.8 * ph / H])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 1, 2])
    cb.ax.tick_params(labelsize=fs - 0.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_label("/ brain median", fontsize=fs - 0.5, labelpad=2)
    fig.savefig(path, dpi=400)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=str(ROOT / "outputs"))
    p.add_argument("--data-root", default=str(ROOT.parent / "HemoPIC" / "data" / "ISLES2017_Training"))
    p.add_argument("--patient", type=int, default=13)
    p.add_argument("--maps-width", type=float, default=3.1, help="width of the map figure in inches")
    p.add_argument("--wide", action="store_true", help="also the one-row map figure for a two-column-wide float")
    a = p.parse_args()
    out, data_root = Path(a.out_root), Path(a.data_root)
    rows = list(csv.DictReader(open(RES / "central_volume.csv", encoding="utf8")))
    vox = {int(r["patient"]): {m: float(r[f"twin-vox tank {m}"]) for m, _ in MAPS} for r in rows}
    hem = {}
    for pid in vox:
        base = data_root / f"training_{pid}"
        brain = load_nifti(base / SEG4_REL)[1] > 0
        ot = load_nifti(find_nii(base, "OT"))[1] > 0
        hem[pid] = aucs(hemopic(out, f"training_{pid}"), brain, ot)
    stats = {"n": len(vox)}
    for m, _ in MAPS:
        d = np.array([vox[p][m] - hem[p][m] for p in vox])
        stats[m] = {"HemoPIC": float(np.mean([hem[p][m] for p in vox])), "twin-vox": float(np.mean([vox[p][m] for p in vox])),
                    "diff": float(d.mean()), "p": float(wilcoxon(d).pvalue), "n_better": int((d > 0).sum())}
    dm = np.array([np.mean([vox[p][m] for m, _ in MAPS]) - np.mean([hem[p][m] for m, _ in MAPS]) for p in vox])
    stats["mean"] = {"diff": float(dm.mean()), "p": float(wilcoxon(dm).pvalue), "n_better": int((dm > 0).sum())}
    (RES / "hemopic_vs_twin.json").write_text(json.dumps(stats, indent=2), encoding="utf8")

    fig_maps(out, data_root, a.patient, vox[a.patient], hem[a.patient], FIG / "hemopic_vs_twin_maps.png", a.maps_width)
    if a.wide:
        fig_maps_wide(out, data_root, a.patient, vox[a.patient], hem[a.patient], FIG / "hemopic_vs_twin_maps_wide.png")

    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.3), sharex=True, sharey=True)
    for k, (m, _) in enumerate(MAPS):
        ax = axes[k]
        ax.plot([0.3, 1.0], [0.3, 1.0], color=INK2, lw=0.6, ls="--")
        ax.scatter([hem[p][m] for p in vox], [vox[p][m] for p in vox], s=6, color=C[k], lw=0, alpha=0.9)
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.3, 1.0)
        ax.set_aspect("equal")
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.tick_params(length=2, pad=1)
        ax.set_title(m, pad=2, color=INK)
        ax.text(0.04, 0.96, f"{stats[m]['n_better']}/{stats['n']} above", transform=ax.transAxes, ha="left", va="top", fontsize=6, color=INK2)
    axes[0].set_ylabel("AUC, twin-vox", labelpad=1)
    axes[1].set_xlabel("AUC, HemoPIC", labelpad=1)
    fig.subplots_adjust(left=0.12, right=0.965, bottom=0.2, top=0.88, wspace=0.14)
    fig.savefig(FIG / "hemopic_vs_twin_auc.png", dpi=400)
    plt.close(fig)
    print(json.dumps(stats, indent=1))
    print("figures ->", FIG / "hemopic_vs_twin_maps.png", FIG / "hemopic_vs_twin_auc.png")


if __name__ == "__main__":
    main()
