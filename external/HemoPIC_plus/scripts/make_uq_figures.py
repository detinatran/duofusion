"""Figures for the uncertainty study (run after the UQ scripts).

python scripts/make_uq_figures.py --data-root ../HemoPIC/data/ISLES2017_Training [--configs baseline full full_aif]
Writes docs/figures/uq_*.png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import C, INK, INK2  # noqa: E402

FIG = ROOT / "docs" / "figures"
LABEL = {"baseline": "HemoPIC original", "full": "HemoPIC+ full", "full_aif": "HemoPIC+ full_aif",
         "full__npde": "neural PDE full", "full_aif__npde": "neural PDE full_aif"}
PARAMS = [("log_R", "R"), ("log_tau", "τ"), ("log_V", "V"), ("log_F", "F"), ("log_Cin", "C_in"), ("delay", "delay"),
          ("log_disp", "dispersion"), ("log_Pe", "Pe")]
SEQ = ["#2a78d6", "#8fb8ea", "#d7e5f6", "#b9b8b3"]
PREFIX = "uq_"


def rows_of(out, cfg):
    rows = []
    for f in sorted((out / cfg).glob("training_*/uq/roi_posterior.csv")):
        with open(f, encoding="utf8") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def fig_identifiability(out, cfgs):
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.0 * len(cfgs), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cfg in zip(axes, cfgs):
        rows = rows_of(out, cfg)
        y = np.arange(len(PARAMS))[::-1]
        for j, (p, lab) in enumerate(PARAMS):
            if f"{p}_post_sd" not in rows[0]:
                ax.text(0.5, y[j], "not in this model", ha="center", va="center", fontsize=7.5, color=INK2)
                continue
            pinned = np.array([r[f"{p}_pinned"] == "True" for r in rows])
            ratio = np.array([float(r[f"{p}_post_sd"]) / max(float(r[f"{p}_prior_sd"]), 1e-30) for r in rows])
            fr = [np.mean(~pinned & (ratio < 0.5)), np.mean(~pinned & (ratio >= 0.5) & (ratio <= 0.9)),
                  np.mean(~pinned & (ratio > 0.9)), np.mean(pinned)]
            left = 0.0
            for k, v in enumerate(fr):
                ax.barh(y[j], v, left=left, color=SEQ[k], height=0.7, edgecolor="#fcfcfb", linewidth=1.5)
                if v > 0.12:
                    ax.text(left + v / 2, y[j], f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                            color="#fcfcfb" if k == 0 else INK)
                left += v
        ax.set_yticks(y, [lab for _, lab in PARAMS])
        ax.set_xlim(0, 1)
        ax.set_title(LABEL.get(cfg, cfg), loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEQ]
    fig.legend(handles, ["learnt from data (sd ratio < 0.5)", "partly (0.5–0.9)", "prior decides (> 0.9)", "stuck at a bound"],
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("What does the data actually determine? Posterior sd / prior sd per ROI parameter", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / f"{PREFIX}identifiability.png", dpi=160)
    plt.close(fig)


def fig_calibration(out, cfgs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot([0.4, 1], [0.4, 1], color=INK2, lw=0.8, ls="--")
    for i, cfg in enumerate(cfgs):
        f = out / cfg / "eval_uq" / "heldout.csv"
        if not f.exists():
            continue
        with open(f, encoding="utf8") as fh:
            rows = list(csv.DictReader(fh))
        levels = sorted({float(r["level"]) for r in rows})
        cov = []
        for lev in levels:
            sel = [r for r in rows if float(r["level"]) == lev]
            n = np.array([float(r["n"]) for r in sel])
            cov.append(np.sum(n * np.array([float(r["covered"]) for r in sel])) / n.sum())
        ax.plot(levels, cov, "o-", color=C[i], lw=2, ms=6, label=LABEL.get(cfg, cfg))
    ax.set_xlabel("nominal interval")
    ax.set_ylabel("hidden frames covered")
    ax.set_title("Check 1: hidden time frames", loc="left", color=INK)
    ax.legend(fontsize=8)
    ax = axes[1]
    qty = ["CBF", "V", "MTT", "tau"]
    w = 0.25
    for i, cfg in enumerate(cfgs):
        f = out / cfg / "eval_uq" / "semisynth.csv"
        if not f.exists():
            continue
        with open(f, encoding="utf8") as fh:
            rows = list(csv.DictReader(fh))
        cov = [np.mean([r[f"{q}_covered"] == "True" for r in rows]) for q in qty]
        x = np.arange(len(qty)) + (i - (len(cfgs) - 1) / 2) * w
        ax.bar(x, cov, w * 0.92, color=C[i], label=LABEL.get(cfg, cfg))
        for xx, v in zip(x, cov):
            ax.text(xx, v + 0.01, f"{v:.0%}", ha="center", fontsize=7, color=INK)
    ax.axhline(0.95, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(range(len(qty)), ["CBF", "CBV (V)", "MTT", "τ"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("truth inside the 95% interval")
    ax.set_title("Check 3: noise added to the fitted curves", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / f"{PREFIX}calibration.png", dpi=160)
    plt.close(fig)


def fig_selective(out, cfgs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=False)
    for ax, m in zip(axes, ("CBF", "MTT")):
        for i, cfg in enumerate(cfgs):
            f = out / cfg / "eval_uq" / "summary.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text(encoding="utf8"))["selective"].get(m)
            if not s:
                continue
            fr = np.array(s["fractions"]) * 100
            ax.plot(fr, s["auc_uncertain_removed"], "o-", color=C[i], lw=2, ms=5)
            ax.plot(fr, s["auc_random_removed"], ":", color=C[i], lw=1.5)
        ax.set_xlabel("% of voxels removed")
        ax.set_ylabel("lesion ROC AUC")
        ax.set_title(f"Check 2: {m}", loc="left", color=INK)
    handles = [plt.Line2D([], [], color=C[i], lw=2) for i in range(len(cfgs))]
    handles += [plt.Line2D([], [], color=INK2, lw=2, marker="o", ms=5), plt.Line2D([], [], color=INK2, lw=1.5, ls=":")]
    fig.legend(handles, [LABEL.get(c, c) for c in cfgs] + ["most uncertain voxels removed", "random voxels removed"],
               loc="lower center", ncol=len(cfgs) + 2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / f"{PREFIX}selective.png", dpi=160)
    plt.close(fig)


def fig_tau(out, cfgs):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    data = {}
    for cfg in cfgs:
        f = out / cfg / "eval_uq" / "tau_diff_draws.npy"
        if f.exists():
            s = json.loads((out / cfg / "eval_uq" / "summary.json").read_text(encoding="utf8"))["tau_lesion_minus_contra"]
            data[cfg] = (np.load(f), s)
    allv = np.concatenate([d for d, _ in data.values()])
    bins = np.linspace(np.floor(allv.min()), np.ceil(allv.max()), 70)
    top = 0
    for i, cfg in enumerate(cfgs):
        if cfg not in data:
            continue
        d, s = data[cfg]
        h, *_ = ax.hist(d, bins=bins, color=C[i], alpha=0.6,
                        label=f"{LABEL.get(cfg, cfg)}: 95% CI [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}] s")
        top = max(top, h.max())
    ax.set_ylim(0, top * 1.45)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("τ(lesion) − τ(contralateral), seconds   (< 0: lesion relaxes faster, the paper's claim)")
    ax.set_ylabel("posterior draws")
    ax.legend(fontsize=8, loc="upper center", ncol=1, frameon=False)
    ax.set_title("The paper's τ claim with uncertainty", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / f"{PREFIX}tau.png", dpi=160)
    plt.close(fig)


def fig_maps(out, data_root, cfgs, pid=22, show_title=True):
    import nibabel as nib

    def load(p):
        return nib.as_closest_canonical(nib.load(str(p))).get_fdata()

    base = Path(data_root) / f"training_{pid}"
    seg = load(base / "hemopic_seg4" / "seg4_label_ref24.nii.gz") > 0
    ot = load(base / "OT.nii") > 0
    z = int(np.argmax(ot.sum(axis=(0, 1))))
    xs, ys = np.where(seg[:, :, z])
    crop = (slice(xs.min() - 3, xs.max() + 4), slice(ys.min() - 3, ys.max() + 4))
    cols = [("CBF", "CBF / brain median", None), ("CBF", "CBF: posterior 95% width / value", "uq"),
            ("CBF", "CBF: ensemble sd / value" if any("__npde" in c for c in cfgs) else "CBF: pipeline sd / value", "ens"), ("MTT", "MTT: posterior 95% width / value", "uq")]
    fig, axes = plt.subplots(len(cfgs), len(cols), figsize=(3.0 * len(cols), 2.9 * len(cfgs)), squeeze=False)
    for i, cfg in enumerate(cfgs):
        fd = out / cfg / f"training_{pid}"
        for j, (m, title, kind) in enumerate(cols):
            ax = axes[i, j]
            v = load(fd / f"HemoPIC_{m}.nii.gz")
            ok = seg & np.isfinite(v) & (v > 0)
            if kind is None:
                img = np.where(ok, v / np.median(v[ok]), np.nan)
                vmax, cmap = 2.5, "Blues"
            elif kind == "uq":
                lo, hi = load(fd / "uq" / f"HemoPIC_{m}_lo.nii.gz"), load(fd / "uq" / f"HemoPIC_{m}_hi.nii.gz")
                img = np.where(ok, (hi - lo) / v, np.nan)
                vmax, cmap = 1.5, "Oranges"
            else:
                sd = load(fd / "ens" / f"HemoPIC_{m}_ens_sd.nii.gz")
                img = np.where(ok, sd / v, np.nan)
                vmax, cmap = 0.5, "Oranges"
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad(plt.rcParams["axes.facecolor"])
            im = ax.imshow(img[:, :, z][crop].T, origin="lower", cmap=cm, vmin=0, vmax=vmax, interpolation="nearest")
            ax.contour(ot[:, :, z][crop].T, levels=[0.5], colors=[INK], linewidths=0.8, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=8.5, color=INK)
            if j == 0:
                ax.set_ylabel(LABEL.get(cfg, cfg), fontsize=9, color=INK)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if show_title:
        fig.suptitle(f"Uncertainty maps, training_{pid} slice {z} (black contour = lesion)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / f"{PREFIX}maps_example.png", dpi=150)
    plt.close(fig)


def fig_maps_column(out, data_root, cfgs, pid=22, width=3.35, fs=6, labels=None, titles=None, end_ticks=False):
    """Same panels as fig_maps, laid out for one column of a two-column paper:
    no title, white background, one shared colour bar under each column.
    fs: font size in points at the printed width; labels: row names (default LABEL);
    titles: column titles in place of the defaults; end_ticks: colour bars labelled at 0 and max only."""
    import nibabel as nib

    def load(p):
        return nib.as_closest_canonical(nib.load(str(p))).get_fdata()

    base = Path(data_root) / f"training_{pid}"
    seg = load(base / "hemopic_seg4" / "seg4_label_ref24.nii.gz") > 0
    ot = load(base / "OT.nii") > 0
    z = int(np.argmax(ot.sum(axis=(0, 1))))
    xs, ys = np.where(seg[:, :, z])
    crop = (slice(xs.min() - 3, xs.max() + 4), slice(ys.min() - 3, ys.max() + 4))
    ens = "ensemble sd" if any("__npde" in c for c in cfgs) else "pipeline sd"
    cols = [("CBF", "CBF", None, 2.5, "Blues"), ("CBF", "CBF 95% width", "uq", 1.5, "Oranges"),
            ("CBF", f"CBF {ens}", "ens", 0.5, "Oranges"), ("MTT", "MTT 95% width", "uq", 1.5, "Oranges")]
    if titles:
        cols = [(m, t, kind, vmax, cmap) for (m, _, kind, vmax, cmap), t in zip(cols, titles)]
    extra = 0.012 * fs * max(c[1].count("\n") for c in cols)
    rc = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": fs}
    with plt.rc_context(rc):
        h = 0.53 * width + 0.25 + 0.06 * (fs - 6) + extra
        fig = plt.figure(figsize=(width, h))
        gs = fig.add_gridspec(len(cfgs) + 1, len(cols), height_ratios=[1] * len(cfgs) + [0.07],
                              left=0.06 + 0.008 * (fs - 6), right=0.995, top=0.94 - 0.01 * (fs - 6) - extra / h, bottom=0.07 + 0.008 * (fs - 6), wspace=0.04, hspace=0.06)
        for j, (m, title, kind, vmax, cmap) in enumerate(cols):
            im = None
            for i, cfg in enumerate(cfgs):
                fd = out / cfg / f"training_{pid}"
                v = load(fd / f"HemoPIC_{m}.nii.gz")
                ok = seg & np.isfinite(v) & (v > 0)
                if kind is None:
                    img = np.where(ok, v / np.median(v[ok]), np.nan)
                elif kind == "uq":
                    lo, hi = load(fd / "uq" / f"HemoPIC_{m}_lo.nii.gz"), load(fd / "uq" / f"HemoPIC_{m}_hi.nii.gz")
                    img = np.where(ok, (hi - lo) / v, np.nan)
                else:
                    img = np.where(ok, load(fd / "ens" / f"HemoPIC_{m}_ens_sd.nii.gz") / v, np.nan)
                cm = plt.get_cmap(cmap).copy()
                cm.set_bad("white")
                ax = fig.add_subplot(gs[i, j])
                im = ax.imshow(img[:, :, z][crop].T, origin="lower", cmap=cm, vmin=0, vmax=vmax, interpolation="nearest")
                ax.contour(ot[:, :, z][crop].T, levels=[0.5], colors=[INK], linewidths=0.5, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if i == 0:
                    ax.set_title(title, fontsize=fs, pad=2, color=INK)
                if j == 0:
                    ax.set_ylabel((labels or {}).get(cfg, LABEL.get(cfg, cfg)), fontsize=fs, labelpad=1, color=INK)
            cax = fig.add_subplot(gs[len(cfgs), j])
            pos = cax.get_position()
            cax.set_position([pos.x0 + 0.12 * pos.width, pos.y0, 0.76 * pos.width, pos.height])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, vmax] if end_ticks else [0, vmax / 2, vmax], format="%g")
            cb.ax.tick_params(labelsize=fs, length=1.5, pad=1)
            cb.outline.set_linewidth(0.4)
        fig.savefig(FIG / f"{PREFIX}maps_column.png", dpi=400)
        plt.close(fig)


def fig_roi_example(out, data_root, cfgs, pid=22):
    """Tracer curves of the most lesion-overlapping and the most contralateral ROI with
    the 95% band of the fitted curve, and the posterior of (F, V) against the prior."""
    import nibabel as nib
    import torch
    from hemopic_plus.uq import laplace as LP
    from hemopic_plus.uq.data import load_subject
    from hemopic_plus.uq.torch_model import DTYPE, simulate

    ot = nib.load(str(Path(data_root) / f"training_{pid}" / "OT.nii")).get_fdata() > 0
    flip = np.flip(ot, axis=0) & ~ot
    fig, axes = plt.subplots(len(cfgs), 3, figsize=(12, 3.3 * len(cfgs)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    for i, cfg in enumerate(cfgs):
        fd = out / cfg / f"training_{pid}"
        if "__npde" in cfg:
            from hemopic_plus.npde.uqbatch import load_npde_subject
            b = load_npde_subject(out / cfg.split("__")[0] / f"training_{pid}", fd)
        else:
            b = load_subject(fd)
        z = np.load(fd / "uq" / "posterior.npz")
        with open(fd / "uq" / "roi_posterior.csv", encoding="utf8") as fh:
            post = {int(r["roi_id"]): r for r in csv.DictReader(fh)}
        lab = nib.load(str(fd / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
        rid_les = max(z["roi_ids"], key=lambda r: (ot & (lab == r)).sum())
        rid_con = max(z["roi_ids"], key=lambda r: (flip & (lab == r)).sum())
        t = b.t.numpy()
        for j, (rid, name) in enumerate(((rid_les, "lesion ROI"), (rid_con, "contralateral ROI"))):
            ax = axes[i, j]
            k = int(np.where(b.roi_ids == rid)[0][0])
            kz = int(np.where(z["roi_ids"] == rid)[0][0])
            draws = torch.as_tensor(z["theta"][kz][::5], dtype=DTYPE)
            sub = b.subset([k] * draws.shape[0])
            with torch.no_grad():
                _, cd = simulate(sub, draws, roi_index=list(range(draws.shape[0])))
                _, cmap_ = simulate(b.subset([k]), torch.as_tensor(z["theta_map"][kz][None], dtype=DTYPE), roi_index=[0])
            lo, hi = np.percentile(cd.numpy(), [2.5, 97.5], axis=0)
            ax.fill_between(t, lo, hi, color=C[i], alpha=0.25, lw=0, label="95% band of the model curve")
            ax.plot(t, b.c_obs[k].numpy(), "o", ms=2.5, color=INK2, label="measured (smoothed)")
            ax.plot(t, cmap_[0].numpy(), color=C[i], lw=1.6, label="posterior mode")
            r = post[int(rid)]
            txt = "\n".join(f"{q} {float(r[f'{q}_mean']):.3g} [{float(r[f'{q}_lo']):.3g}, {float(r[f'{q}_hi']):.3g}]"
                            for q in ("CBF", "V", "MTT", "tau"))
            ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK, family="monospace")
            ax.set_title(f"{LABEL.get(cfg, cfg)}: {name} ({b.tissue[k]}, roi {rid})", loc="left", fontsize=9, color=INK)
            ax.set_xlabel("time (s)")
            if j == 0:
                ax.set_ylabel("tracer concentration (a.u.)")
                if i == 0:
                    ax.legend(fontsize=7, loc="upper left", frameon=False)
        ax = axes[i, 2]
        k = int(np.where(b.roi_ids == rid_les)[0][0])
        kz = int(np.where(z["roi_ids"] == rid_les)[0][0])
        kappa = torch.as_tensor([float(post[int(rid_les)]["kappa"])], dtype=DTYPE)
        psd = LP.prior_sd(b.subset([k]), kappa)[0].numpy()
        f0, v0 = float(b.log_f0[k]), float(b.log_v0[k])
        th = z["theta"][kz]
        ax.scatter(th[:, 3] - f0, th[:, 2] - v0, s=3, alpha=0.3, color=C[i], lw=0, label="posterior draws")
        ang = np.linspace(0, 2 * np.pi, 200)
        ax.plot(2 * psd[3] * np.cos(ang), 2 * psd[2] * np.sin(ang), color=INK2, lw=1, ls="--", label="prior, 2 sd")
        ax.plot(0, 0, "+", color=INK, ms=8)
        dx, dy = th[:, 3] - f0, th[:, 2] - v0
        xs = np.concatenate([[-2 * psd[3], 2 * psd[3]], np.percentile(dx, [0.5, 99.5])])
        ys = np.concatenate([[-2 * psd[2], 2 * psd[2]], np.percentile(dy, [0.5, 99.5])])
        c, h = (xs.min() + xs.max()) / 2, 0.55 * max(np.ptp(xs), np.ptp(ys))
        cy = (ys.min() + ys.max()) / 2
        ax.set_xlim(c - h, c + h)
        ax.set_ylim(cy - h, cy + h)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("log F − prior centre")
        ax.set_ylabel("log V − prior centre")
        ax.set_title("lesion ROI: F and V vs their prior", loc="left", fontsize=9, color=INK)
        ax.legend(fontsize=7, loc="upper right", frameon=True, facecolor="#fcfcfb", edgecolor="none", markerscale=3)
    fig.suptitle(f"Posterior of single ROIs, training_{pid}", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / f"{PREFIX}roi_example.png", dpi=150)
    plt.close(fig)


def fig_widths(out, cfgs):
    """Median relative 95% width per quantity and tissue, and lesion vs contralateral voxels."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    qty = [("CBF", "CBF"), ("V", "CBV (V)"), ("MTT", "MTT"), ("tau", "τ")]
    w = 0.26
    for i, cfg in enumerate(cfgs):
        s = json.loads((out / cfg / "eval_uq" / "summary.json").read_text(encoding="utf8"))["rel_width_95"]
        x = np.arange(len(qty)) + (i - (len(cfgs) - 1) / 2) * w
        ax.bar(x, [s[q]["all"] for q, _ in qty], w * 0.92, color=C[i], label=LABEL.get(cfg, cfg))
        ax.plot(x, [s[q]["GM"] for q, _ in qty], "^", color=INK, ms=4)
        ax.plot(x, [s[q]["WM"] for q, _ in qty], "v", color=INK2, ms=4)
        for xx, (q, _) in zip(x, qty):
            ax.text(xx, max(s[q]["all"], s[q]["GM"]) + 0.06, f"{s[q]['all']:.0%}", ha="center", fontsize=7, color=INK)
    ax.plot([], [], "^", color=INK, ms=4, label="grey matter")
    ax.plot([], [], "v", color=INK2, ms=4, label="white matter")
    ax.set_xticks(range(len(qty)), [label for _, label in qty])
    ax.set_ylabel("95% interval width / value (ROI median)")
    ax.set_title("How wide are the intervals? (651 ROIs)", loc="left", color=INK)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left")
    ax.set_ylim(0, 3.1)
    ax = axes[1]
    for i, cfg in enumerate(cfgs):
        s = json.loads((out / cfg / "eval_uq" / "summary.json").read_text(encoding="utf8"))["selective"]
        for j, m in enumerate(("CBF", "MTT")):
            x = j + (i - (len(cfgs) - 1) / 2) * w
            ax.plot([x, x], [s[m]["u_contra_median"], s[m]["u_lesion_median"]], color=C[i], lw=2)
            ax.plot(x, s[m]["u_contra_median"], "o", color=C[i], mfc="#fcfcfb", ms=6)
            ax.plot(x, s[m]["u_lesion_median"], "o", color=C[i], ms=6)
    ax.plot([], [], "o", color=INK2, ms=6, label="lesion")
    ax.plot([], [], "o", color=INK2, mfc="#fcfcfb", ms=6, label="contralateral")
    ax.set_xticks([0, 1], ["CBF", "MTT"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("voxel 95% width / value (patient median)")
    ax.set_title("Lesion vs contralateral voxels", loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"{PREFIX}widths.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=str(ROOT / "outputs"))
    p.add_argument("--data-root", default=str(ROOT.parent / "HemoPIC" / "data" / "ISLES2017_Training"))
    p.add_argument("--configs", nargs="*", default=["baseline", "full", "full_aif"])
    p.add_argument("--map-configs", nargs="*", help="rows of the map and ROI figures (default: configs without full_aif)")
    p.add_argument("--prefix", default="uq_", help="figure file prefix, e.g. uq_npde_")
    a = p.parse_args()
    global PREFIX
    PREFIX = a.prefix
    out = Path(a.out_root)
    FIG.mkdir(parents=True, exist_ok=True)
    fig_identifiability(out, a.configs)
    fig_calibration(out, a.configs)
    fig_selective(out, a.configs)
    fig_tau(out, a.configs)
    mc = a.map_configs or [c for c in a.configs if c != "full_aif"]
    fig_maps(out, a.data_root, mc)
    fig_maps_column(out, a.data_root, mc)
    fig_roi_example(out, a.data_root, mc)
    fig_widths(out, a.configs)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
