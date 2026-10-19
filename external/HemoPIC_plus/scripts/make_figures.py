"""Figures for the README / docs.

python scripts/make_figures.py                               # results figures (docs/results only)
python scripts/make_figures.py --outputs outputs --data-root ../HemoPIC/data/ISLES2017_Training
                                                             # + diagnosis, tracer curves, example maps

Writes docs/figures/*.png.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RES = ROOT / "docs" / "results"
FIG = ROOT / "docs" / "figures"
MAPS = ("CBF", "CBV", "MTT")

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "legend.frameon": False,
})


def roc_table(exp, method="model"):
    d = {}
    with open(RES / exp / "per_patient.csv", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["kind"] == "roc" and r["method"] == method and r["value"] not in ("", "nan"):
                d[(int(r["patient"]), r["region"])] = float(r["value"])
    return d


def tracer_table(exp):
    d = {}
    with open(RES / exp / "per_patient.csv", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["kind"] == "tracer" and r["value"] not in ("", "nan"):
                d[(int(r["patient"]), r["region"])] = float(r["value"])
    return d


def boot_ci(v, n=4000, seed=0):
    v = np.asarray(v)
    m = v[np.random.default_rng(seed).integers(0, v.size, (n, v.size))].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def fig_auc_36():
    """Lesion AUC of five methods on the 36 subjects with an accepted AIF."""
    osvd = {}
    with open(RES / "osvd" / "roc.csv", encoding="utf8") as f:
        for r in csv.DictReader(f):
            osvd[(int(r["patient"]), r["map"])] = float(r["auc"])
    pids = sorted({k[0] for k in osvd})
    methods = [("Deconv (ISLES)", roc_table("baseline", "deconv")), ("HemoPIC (original)", roc_table("baseline")),
               ("oSVD + auto AIF", osvd), ("HemoPIC+ full_idreg", roc_table("full_idreg")),
               ("HemoPIC+ full_aif_idvox_osvd_nc", roc_table("full_aif_idvox_osvd_nc"))]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    w, gap = 0.15, 0.012
    for i, (name, d) in enumerate(methods):
        for j, m in enumerate(MAPS):
            v = np.array([d[(p, m)] for p in pids if (p, m) in d])
            x = j + (i - 2) * (w + gap)
            lo, hi = boot_ci(v)
            ax.bar(x, v.mean() - 0.5, w, bottom=0.5, color=C[i], label=name if j == 0 else None, linewidth=0)
            ax.plot([x, x], [lo, hi], color=INK2, lw=1)
            ax.text(x, hi + 0.004, f"{v.mean():.3f}", ha="center", va="bottom", fontsize=7, color=INK, rotation=90)
    ax.set_xticks(range(3), [f"{m}" for m in MAPS])
    ax.set_ylim(0.5, 0.87)
    ax.set_ylabel("Lesion ROC AUC (mean over patients)")
    ax.set_title(f"Lesion discrimination on the same {len(pids)} patients (95% bootstrap CI)", loc="left", color=INK)
    ax.legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.08))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "auc_comparison.png", dpi=160)
    plt.close(fig)


def fig_ablation():
    """Cumulative ablation over all 43 patients: three AUCs and lesion nRMSE."""
    steps = [("baseline", "HemoPIC original"), ("c1_tr", "+ C1 true TR"), ("c15_units", "+ C5 flow units"),
             ("c152_delay", "+ C2 bolus delay"), ("full_ref", "+ C4 multistart"), ("full", "+ C3 reference-free"),
             ("full_aif", "+ C6 auto AIF"), ("full_aif_idreg", "+ C8 ID regional CBV"),
             ("full_aif_idvox_osvd_nc", "+ C8/C9 ID voxel CBV, oSVD CBF")]
    deconv = roc_table("baseline", "deconv")
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.9), sharey=True)
    y = np.arange(len(steps))[::-1]
    for k, m in enumerate(MAPS + ("nRMSE lesion",)):
        ax = axes[k]
        vals = []
        for exp, _ in steps:
            if m == "nRMSE lesion":
                d = tracer_table(exp)
                vals.append(np.mean([v for (p, r), v in d.items() if r == "lesion"]))
            else:
                d = roc_table(exp)
                vals.append(np.mean([v for (p, r), v in d.items() if r == m]))
        ax.plot(vals, y, color=C[0], lw=2, zorder=2)
        ax.scatter(vals, y, s=36, color=C[0], edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        if m != "nRMSE lesion":
            ref = np.mean([v for (p, r), v in deconv.items() if r == m])
            ax.axvline(ref, color=C[1], lw=1.5, ls="--", zorder=1)
            ax.text(ref, y[0] + 0.75, "Deconv", color=INK2, fontsize=8, ha="center")
            ax.set_title(f"AUC {m}  (higher better)", loc="left", fontsize=10, color=INK)
        else:
            ax.set_title("nRMSE lesion  (lower better)", loc="left", fontsize=10, color=INK)
        span = max(vals) - min(vals)
        for yy, vv in ((y[0], vals[0]), (y[-1], vals[-1])):
            ax.annotate(f"{vv:.3f}", (vv, yy), xytext=(7, -3), textcoords="offset points", fontsize=8, color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none"))
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.18 * span)
        ax.set_ylim(-0.8, len(steps) + 0.4)
    axes[0].set_yticks(y, [s[1] for s in steps])
    fig.suptitle("Cumulative ablation (43 patients)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=160)
    plt.close(fig)


def fig_prior_diagnosis(outputs):
    """Fitted F and V relative to their priors: original model vs unit-consistent model."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    bins = np.logspace(np.log10(0.2), np.log10(5), 41)
    for ax, key, lab in ((axes[0], ("F_fit", "F_prior"), "F fitted / F prior"), (axes[1], ("V_fit", "V_prior"), "V fitted / V prior")):
        for i, (exp, name) in enumerate((("baseline", "HemoPIC original"), ("full_ref", "HemoPIC+ (C1+C5+C2+C4)"))):
            r = []
            for f in Path(outputs, exp).glob("training_*/roi_fit_summary.csv"):
                with open(f, encoding="utf8") as fh:
                    r += [float(x[key[0]]) / float(x[key[1]]) for x in csv.DictReader(fh) if float(x[key[1]]) > 0]
            ax.hist(np.clip(r, 0.2, 5), bins=bins, color=C[i], alpha=0.85 if i else 1.0, label=name,
                    histtype="stepfilled" if i == 0 else "step", lw=2)
        ax.set_xscale("log")
        ax.set_xticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(lab)
        ax.axvline(1, color=INK2, lw=0.8)
    axes[0].set_ylabel("ROIs (43 patients)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Original HemoPIC leaves F and V at their priors (taken from the Deconv maps)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "prior_diagnosis.png", dpi=160)
    plt.close(fig)


def fig_tracer_curves(outputs, data_root):
    """Cohort-mean observed vs fitted lesion / contralateral curves (cf. paper Fig. 5b)."""
    from hemopic_plus.io import find_nii, load_nifti, squeeze_ctc
    from hemopic_plus.signal import compute_peak_map, smooth_curve_gaussian
    import json
    exps = (("baseline", "HemoPIC original"), ("full", "HemoPIC+ (full, same fits as full_idreg)"))
    tr = {int(k): v for k, v in json.loads((ROOT / "configs" / "isles2017_tr.json").read_text()).items()}
    t_grid = np.arange(0, 60.01, 1.0)
    acc = defaultdict(list)
    for exp, _ in exps:
        for d in sorted(Path(outputs, exp).glob("training_*")):
            pid = int(d.name.split("_")[1])
            base = Path(data_root) / d.name
            info = json.loads((d / "run_info_core.json").read_text(encoding="utf8"))
            ctc = squeeze_ctc(load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))[1])
            sub3 = load_nifti(d / "roi_labels_seg4_within_kmeans.nii.gz")[1].astype(int)
            ot = load_nifti(find_nii(base, "OT"))[1] > 0
            show = (sub3 > 0) & np.all(np.isfinite(ctc), axis=3)
            fm = show & (compute_peak_map(ctc, info["baseline_frames"]) > info["aif_peak_threshold"])
            z = np.load(d / "fit_curves.npz")
            curves = {int(r): z["c_fit"][i] for i, r in enumerate(z["roi_ids"]) if z["success"][i]}
            t = np.arange(ctc.shape[3]) * tr.get(pid, 1.44)
            for reg, m in (("lesion", ot & fm), ("c-lesion", np.flip(ot, 0) & fm)):
                if m.sum() < 10:
                    continue
                obs = smooth_curve_gaussian(ctc[m].mean(0), 1.8, info["baseline_frames"], True)
                num, den = np.zeros(ctc.shape[3]), 0
                for rid, cf in curves.items():
                    a = int((m & (sub3 == rid)).sum())
                    num, den = num + a * cf, den + a
                if den == 0:
                    continue
                acc[(exp, reg, "obs")].append(np.interp(t_grid, t, obs))
                acc[(exp, reg, "fit")].append(np.interp(t_grid, t, num / den))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (exp, name) in zip(axes, exps):
        for k, reg in enumerate(("lesion", "c-lesion")):
            col = C[1] if reg == "lesion" else C[0]
            ax.plot(t_grid, np.mean(acc[(exp, reg, "obs")], 0), color=col, lw=2, label=f"{reg} observed")
            ax.plot(t_grid, np.mean(acc[(exp, reg, "fit")], 0), color=col, lw=2, ls="--", label=f"{reg} fitted")
        ax.set_title(name, loc="left", color=INK)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("concentration (a.u.)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Cohort-mean tracer reconstruction, lesion vs contralateral (43 patients)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "tracer_curves.png", dpi=160)
    plt.close(fig)


def fig_maps(outputs, data_root, pid=22):
    """Relative perfusion maps (divided by the brain median) of one patient."""
    import nibabel as nib
    base = Path(data_root) / f"training_{pid}"

    def load(p):
        return nib.as_closest_canonical(nib.load(str(p))).get_fdata()

    seg = load(base / "hemopic_seg4" / "seg4_label_ref24.nii.gz") > 0
    ot = load(base / "OT.nii") > 0
    z = int(np.argmax(ot.sum(axis=(0, 1))))
    rows = [("Deconv (ISLES)", {m: base / f"MR_{'r' if m != 'MTT' else ''}{m}.nii" for m in MAPS}),
            ("HemoPIC original", {m: Path(outputs, "baseline", f"training_{pid}", f"HemoPIC_{m}.nii.gz") for m in MAPS}),
            ("oSVD + auto AIF", {m: Path(outputs, "osvd", f"training_{pid}", f"oSVD_{m}.nii.gz") for m in MAPS}),
            ("HemoPIC+ full_aif_idvox_osvd_nc", {m: Path(outputs, "full_aif_idvox_osvd_nc", f"training_{pid}", f"HemoPIC_{m}.nii.gz") for m in MAPS})]
    xs, ys = np.where(seg[:, :, z])
    crop = (slice(xs.min() - 3, xs.max() + 4), slice(ys.min() - 3, ys.max() + 4))
    fig, axes = plt.subplots(len(rows), 4, figsize=(9.6, 2.45 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#fcfcfb")
    for i, (name, paths) in enumerate(rows):
        for j, m in enumerate(MAPS):
            v = load(paths[m])
            ok = seg & np.isfinite(v) & (v > 0)
            rel = np.where(ok, v / np.median(v[ok]), np.nan)[:, :, z][crop]
            ax = axes[i, j]
            ax.imshow(rel.T, origin="lower", cmap=cmap, vmin=0, vmax=2.5, interpolation="nearest")
            ax.contour(ot[:, :, z][crop].T, levels=[0.5], colors=[C[1]], linewidths=1.0, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f"relative {m}", color=INK)
            if j == 0:
                ax.set_ylabel(name, color=INK, fontsize=9)
        ax = axes[i, 3]
        ax.imshow(ot[:, :, z][crop].T, origin="lower", cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if i == 0:
            ax.set_title("lesion (OT)", color=INK)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 2.5))
    cb = fig.colorbar(sm, ax=axes[:, :3], orientation="horizontal", fraction=0.025, pad=0.02)
    cb.set_label("value / brain median (orange contour = lesion)", color=INK2)
    fig.suptitle(f"Example: training_{pid}, slice {z}", x=0.01, ha="left", color=INK)
    fig.savefig(FIG / "maps_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", help="out_root with experiment folders (enables image-based figures)")
    p.add_argument("--data-root")
    a = p.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_auc_36()
    fig_ablation()
    if a.outputs:
        fig_prior_diagnosis(a.outputs)
        if a.data_root:
            fig_tracer_curves(a.outputs, a.data_root)
            fig_maps(a.outputs, a.data_root)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
