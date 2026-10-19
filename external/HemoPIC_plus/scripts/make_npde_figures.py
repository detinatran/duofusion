"""Figures for the transport PDE / neural PDE study (run after train_npde.py).

python scripts/make_npde_figures.py [--configs full full_aif] [--data-root ...]
Writes docs/figures/npde_*.png
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
import nibabel as nib  # noqa: E402
import torch  # noqa: E402
from make_figures import C, INK, INK2  # noqa: E402

from hemopic_plus.io import find_nii  # noqa: E402
from hemopic_plus.npde import transport as TP  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE  # noqa: E402

FIG = ROOT / "docs" / "figures"
LABEL = {"full": "HemoPIC+ full", "full_aif": "HemoPIC+ full_aif"}
VAR = [("", "tank (HemoPIC+)", C[0]), ("__pde", "PDE", C[1]), ("__npde", "neural PDE", C[2])]
GROUPS = [("GM", "healthy grey matter"), ("WM", "healthy white matter"), ("lesion", "ROIs > 20% lesion")]
GRID = np.arange(-20, 41, 2.0)


def roi_groups(fd, data_root):
    lab = nib.load(str(fd / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
    ot = nib.load(str(find_nii(data_root / fd.name, "OT"))).get_fdata() > 0
    tis = {int(r["roi_id"]): r["tissue"] for r in csv.DictReader(open(fd / "roi_fit_summary.csv", encoding="utf8"))}
    out = {}
    for rid, t in tis.items():
        m = lab == rid
        frac = (ot & m).sum() / max(m.sum(), 1)
        out[rid] = "lesion" if frac > 0.2 else (t if frac == 0 and t in ("GM", "WM") else None)
    return out


def residuals(out, data_root, cfg):
    res = {(v, g): [] for v, _, _ in VAR for g, _ in GROUPS}
    for fd in sorted((out / cfg).glob("training_*")):
        if not (out / f"{cfg}__npde" / fd.name / "fit_curves.npz").exists():
            continue
        grp = roi_groups(fd, data_root)
        for v, _, _ in VAR:
            z = np.load(out / f"{cfg}{v}" / fd.name / "fit_curves.npz")
            t = z["t"]
            for i, rid in enumerate(z["roi_ids"]):
                g = grp.get(int(rid))
                if g is None or not z["success"][i]:
                    continue
                co, cf = z["c_obs"][i], z["c_fit"][i]
                pk = int(np.argmax(co))
                res[(v, g)].append(np.interp(t[pk] + GRID, t, (co - cf) / co.max(), left=np.nan, right=np.nan))
    return res


def fig_residuals(out, data_root, cfgs):
    fig, axes = plt.subplots(len(cfgs), len(GROUPS), figsize=(12, 3.2 * len(cfgs)), sharex=True, squeeze=False)
    for i, cfg in enumerate(cfgs):
        res = residuals(out, data_root, cfg)
        for j, (g, gl) in enumerate(GROUPS):
            ax = axes[i, j]
            ax.axhline(0, color=INK2, lw=0.8)
            for v, vl, col in VAR:
                a = np.array(res[(v, g)])
                if not a.size:
                    continue
                m = np.nanmean(a, 0)
                se = np.nanstd(a, 0) / np.sqrt(np.sum(np.isfinite(a), 0))
                ax.fill_between(GRID, m - 2 * se, m + 2 * se, color=col, alpha=0.2, lw=0)
                ax.plot(GRID, m, color=col, lw=1.8, label=vl)
            ax.set_title(f"{LABEL.get(cfg, cfg)}: {gl} (n={len(res[('', g)])})", loc="left", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel("mean (measured − fit) / peak")
            if i == len(cfgs) - 1:
                ax.set_xlabel("seconds from the measured peak")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Systematic misfit around the bolus peak (band: ±2 standard errors)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "npde_residuals.png", dpi=150)
    plt.close(fig)


def residuals_rescaled(out, data_root, cfg, ref):
    """residuals() for a fit with a wrong frame interval (HemoPIC's 1 s), with its time
    axis rescaled to the true interval of the fit `ref` of the same subject."""
    res = {g: [] for g, _ in GROUPS}
    for fd in sorted((out / cfg).glob("training_*")):
        zr = out / ref / fd.name / "fit_curves.npz"
        if not zr.exists() or not (fd / "fit_curves.npz").exists():
            continue
        grp = roi_groups(fd, data_root)
        z = np.load(fd / "fit_curves.npz")
        scale = float(np.load(zr)["dt"]) / float(z["dt"])
        t = (z["t"] - z["t"][0]) * scale
        for i, rid in enumerate(z["roi_ids"]):
            g = grp.get(int(rid))
            if g is None or not z["success"][i]:
                continue
            co, cf = z["c_obs"][i], z["c_fit"][i]
            pk = int(np.argmax(co))
            res[g].append(np.interp(t[pk] + GRID, t, (co - cf) / co.max(), left=np.nan, right=np.nan))
    return res


def fig_residuals_paper(out, data_root, cfg="full_aif", baseline="baseline", path=None, fs=12, size=(12, 2.55), legend_panel=0, dpi=200):
    """Paper Fig. 2: one row, HemoPIC (`baseline`) and the three HemoPIC+ tissue models of `cfg`.
    size: inches; at the printed size (e.g. 5.18 x 1.10) fs is the printed font size."""
    var = [("", "HemoPIC+, compartment", C[0]), ("__pde", "HemoPIC+, PDE", C[1]), ("__npde", "HemoPIC+, neural PDE", C[2])]
    global VAR
    VAR = var
    res = residuals(out, data_root, cfg)
    base = residuals_rescaled(out, data_root, baseline, cfg) if baseline else None
    with plt.rc_context({"font.size": fs, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(1, len(GROUPS), figsize=size, sharex=True)
        for j, (g, gl) in enumerate(GROUPS):
            ax = axes[j]
            ax.axhline(0, color=INK2, lw=0.8)
            curves = ([("HemoPIC", INK2, np.array(base[g]), "--")] if base else []) + \
                     [(vl, col, np.array(res[(v, g)]), "-") for v, vl, col in var]
            for lab, col, a, ls in curves:
                m = np.nanmean(a, 0)
                se = np.nanstd(a, 0) / np.sqrt(np.sum(np.isfinite(a), 0))
                ax.fill_between(GRID, m - 2 * se, m + 2 * se, color=col, alpha=0.15, lw=0)
                ax.plot(GRID, m, color=col, lw=1.8 * min(1.0, size[0] / 12) ** 0.5, ls=ls, label=lab)
            n = len(res[('', g)])
            nb = f", HemoPIC n={len(base[g])}" if base and len(base[g]) != n else ""
            ax.set_title(f"{gl} (n={n}{nb})", loc="left", fontsize=fs, color=INK)
            ax.set_xlabel("seconds from the measured peak", fontsize=fs - 1)
        axes[0].set_ylabel("(meas. \u2212 fit) / peak", fontsize=fs - 1, labelpad=1)
        for ax in axes:
            ax.tick_params(labelsize=fs - 1, length=2, pad=1)
        axes[legend_panel].legend(fontsize=fs - 1, frameon=False, handlelength=1.6, labelspacing=0.2, borderaxespad=0.2,
                                  loc="upper right" if legend_panel else "best")
        fig.tight_layout(pad=0.4, w_pad=0.6)
        fig.savefig(path or FIG / "twin_residuals_paper.png", dpi=dpi)
        plt.close(fig)


def fig_residuals_column(out, data_root, cfg="full_aif", baseline="baseline", path=None, fs=6.5, size=(3.39, 1.1), dpi=600, grid=False):
    """Paper Fig. 2 for one column: size is the printed size in inches, fs the printed font size;
    grid puts the three panels in a 2x2 grid with the legend in the fourth cell."""
    var = [("", "compartment", C[0]), ("__pde", "PDE", C[1]), ("__npde", "neural PDE", C[2])]
    global VAR
    VAR = var
    res = residuals(out, data_root, cfg)
    base = residuals_rescaled(out, data_root, baseline, cfg) if baseline else None
    titles = {"GM": "GM", "WM": "WM", "lesion": "lesion ROIs"}
    with plt.rc_context({"font.size": fs, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}):
        if grid:
            fig, axs = plt.subplots(2, 2, figsize=size, sharey=True)
            axes, lax = [axs[0, 0], axs[0, 1], axs[1, 0]], axs[1, 1]
            lax.axis("off")
        else:
            fig, axes = plt.subplots(1, len(GROUPS), figsize=size, sharex=True, sharey=True)
        for ax, (g, _) in zip(axes, GROUPS):
            ax.axhline(0, color=INK2, lw=0.5)
            curves = ([("HemoPIC", INK2, np.array(base[g]), (0, (3, 1.5)))] if base else []) + \
                     [(vl, col, np.array(res[(v, g)]), "-") for v, vl, col in var]
            for lab, col, a, ls in curves:
                m = np.nanmean(a, 0)
                se = np.nanstd(a, 0) / np.sqrt(np.sum(np.isfinite(a), 0))
                ax.fill_between(GRID, m - 2 * se, m + 2 * se, color=col, alpha=0.18, lw=0)
                ax.plot(GRID, m, color=col, lw=0.9, ls=ls, label=lab)
            ax.set_title(f"{titles[g]} (n={len(res[('', g)])})", fontsize=fs, color=INK, pad=2)
            ax.set_xticks([-20, 0, 20, 40])
            ax.tick_params(labelsize=fs, length=2, pad=1)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        h, lab = axes[0].get_legend_handles_labels()
        if grid:
            axes[2].set_xlabel("seconds from the peak", fontsize=fs, labelpad=1)
            axes[1].tick_params(labelleft=True)
            fig.supylabel("residual / peak", fontsize=fs, x=0.01)
            fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.5, rect=(0.035, 0, 1, 1))
            wm, les = axes[1].get_position(), axes[2].get_position()
            fig.legend(h, lab, loc="center", bbox_to_anchor=((wm.x0 + wm.x1) / 2, (les.y0 + les.y1) / 2),
                       bbox_transform=fig.transFigure, fontsize=fs, frameon=False, handlelength=2.0, labelspacing=0.45)
        else:
            axes[0].set_ylabel("residual / peak", fontsize=fs, labelpad=1)
            fig.supxlabel("seconds from the measured peak", fontsize=fs, y=0.0)
            fig.legend(h, lab, loc="upper center", ncol=len(lab), fontsize=fs, frameon=False, handlelength=1.6,
                       columnspacing=0.9, handletextpad=0.4, borderaxespad=0.1)
            fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0, 1, 0.86))
        fig.savefig(path or FIG / "twin_residuals_column.png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def fig_examples(out, data_root, cfg, pid=22):
    fd = out / cfg / f"training_{pid}"
    grp = roi_groups(fd, data_root)
    lab = nib.load(str(fd / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
    ot = nib.load(str(find_nii(data_root / fd.name, "OT"))).get_fdata() > 0
    flip = np.flip(ot, axis=0) & ~ot
    zs = {v: np.load(out / f"{cfg}{v}" / fd.name / "fit_curves.npz") for v, _, _ in VAR}
    ids = [int(r) for r in zs[""]["roi_ids"]]
    les = max(ids, key=lambda r: (ot & (lab == r)).sum())
    con = max(ids, key=lambda r: (flip & (lab == r)).sum())
    gm = [r for r in ids if grp.get(r) == "GM"]
    gm = max(gm, key=lambda r: (lab == r).sum()) if gm else ids[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (rid, name) in zip(axes, ((les, "lesion ROI"), (con, "contralateral ROI"), (gm, "largest healthy GM ROI"))):
        k = ids.index(rid)
        t = zs[""]["t"]
        ax.plot(t, zs[""]["c_obs"][k], "o", ms=2.8, color=INK2, label="measured (smoothed)")
        for v, vl, col in VAR:
            ax.plot(t, zs[v]["c_fit"][k], color=col, lw=1.6, label=vl)
        ax.set_title(f"{name} (roi {rid})", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("tracer concentration (a.u.)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{LABEL.get(cfg, cfg)}, training_{pid}: held-out fits (the network never saw this patient)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / f"npde_examples_{cfg}.png", dpi=150)
    plt.close(fig)


def fig_fields(out, data_root, cfgs, n_cells=24):
    """What the closure learned: split of volume / flow between the two tubes, the
    resulting transit-time spread (CV = CTH / MTT; tank = 1), and the transit-time
    density at the median parameters of each group."""
    gcol = {"GM": C[0], "WM": C[3], "lesion": C[4]}
    gshort = {"GM": "GM", "WM": "WM", "lesion": "lesion ROIs"}
    s_grid = np.linspace(0.0, 4.0, 161)
    fig, axes = plt.subplots(len(cfgs), 3, figsize=(14, 3.6 * len(cfgs)), squeeze=False)
    for i, cfg in enumerate(cfgs):
        coll = {g: {"ratio": [], "cvn": [], "cvp": [], "pe": [], "pep": [], "nu": [], "phif": []} for g, _ in GROUPS}
        for fd in sorted((out / cfg).glob("training_*")):
            f = out / f"{cfg}__npde" / fd.name / "npde_theta.npz"
            if not f.exists():
                continue
            z = np.load(f)
            grp = roi_groups(fd, data_root)
            for k, rid in enumerate(z["roi_ids"]):
                g = grp.get(int(rid))
                if g is None:
                    continue
                nu, phi = z["nu"][k], z["phi"][k]
                fast = int(np.argmax(phi / nu))
                coll[g]["ratio"].append((phi[fast] / nu[fast]) / (phi[1 - fast] / nu[1 - fast]))
                coll[g]["nu"].append(nu[fast])
                coll[g]["phif"].append(phi[fast])
                coll[g]["cvn"].append(z["cv_npde"][k])
                coll[g]["cvp"].append(z["cv_pde"][k])
                coll[g]["pe"].append(z["theta"][k, 7])
                coll[g]["pep"].append(z["theta_pde"][k, 7])
        ax = axes[i, 0]
        for g, gl in GROUPS:
            ax.scatter(coll[g]["phif"], coll[g]["ratio"], s=9, alpha=0.6, color=gcol[g], lw=0, label=gl)
        ax.set_yscale("log")
        ax.set_xlabel("flow fraction of the fast tube")
        ax.set_ylabel("rate ratio fast / slow tube")
        ax.set_title(f"{LABEL.get(cfg, cfg)}: learned split (1 = one tube)", loc="left", fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax = axes[i, 1]
        pos = np.arange(len(GROUPS))
        for vals, off, col in ((([coll[g]["cvp"] for g, _ in GROUPS]), -0.18, C[1]), (([coll[g]["cvn"] for g, _ in GROUPS]), 0.18, C[2])):
            ax.boxplot(vals, positions=pos + off, widths=0.3, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": col, "alpha": 0.6}, medianprops={"color": INK})
        ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
        ax.text(len(GROUPS) - 0.5, 1.0, " tank", va="center", fontsize=7, color=INK2)
        ax.set_xticks(pos, [gshort[g] for g, _ in GROUPS])
        ax.set_ylabel("CV of transit times (CTH / MTT)")
        ax.set_title("transit-time spread", loc="left", fontsize=9, color=INK)
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=C[1], alpha=0.6), plt.Rectangle((0, 0), 1, 1, color=C[2], alpha=0.6)],
                  ["PDE", "neural PDE"], fontsize=7, frameon=False, loc="upper left")
        ax = axes[i, 2]
        top = 0.0
        for g, gl in GROUPS:
            if not coll[g]["pe"]:
                continue
            lp = torch.tensor([float(np.median(coll[g]["pe"]))], dtype=DTYPE)
            lpp = torch.tensor([float(np.median(coll[g]["pep"]))], dtype=DTYPE)
            nu_f, phi_f = float(np.median(coll[g]["nu"])), float(np.median(coll[g]["phif"]))
            h_n = TP.transit_density_paths(lp, n_cells, s_grid, torch.tensor([[nu_f, 1 - nu_f]], dtype=DTYPE),
                                           torch.tensor([[phi_f, 1 - phi_f]], dtype=DTYPE))[0].numpy()
            h_p = TP.transit_density(lpp, n_cells, s_grid)[0].numpy()
            ax.plot(s_grid, h_n, color=gcol[g], lw=1.8)
            ax.plot(s_grid, h_p, color=gcol[g], lw=1.0, ls="--")
            top = max(top, h_n.max(), h_p.max())
        ax.plot(s_grid, np.exp(-s_grid), color=INK2, lw=1.0, ls=":")
        ax.set_ylim(0, 1.45 * max(top, 1.0))
        ax.set_xlabel("transit time / MTT")
        ax.set_ylabel("density")
        ax.set_title("transit-time density at the median parameters", loc="left", fontsize=9, color=INK)
        handles = [plt.Line2D([], [], color=gcol[g], lw=1.8) for g, _ in GROUPS]
        handles += [plt.Line2D([], [], color=INK2, lw=1.8), plt.Line2D([], [], color=INK2, lw=1.0, ls="--"),
                    plt.Line2D([], [], color=INK2, lw=1.0, ls=":")]
        ax.legend(handles, [gshort[g] for g, _ in GROUPS] + ["neural PDE", "PDE", "tank (HemoPIC)"], fontsize=7,
                  frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "npde_fields.png", dpi=150)
    plt.close(fig)


def fig_metrics(out, cfgs):
    tags = [("pipeline", "tank (pipeline)"), ("tank_polished", "tank, polished"), ("pdeadam", "PDE"), ("npde", "neural PDE")]
    fig, axes = plt.subplots(len(cfgs), 2, figsize=(12, 3.2 * len(cfgs)), squeeze=False, gridspec_kw={"width_ratios": [1.3, 1]})
    for i, cfg in enumerate(cfgs):
        s = json.loads((out / f"{cfg}__npde" / "eval" / "summary.json").read_text(encoding="utf8"))
        for j, (keys, names, title) in enumerate((
                (["nRMSE_GM", "nRMSE_WM", "nRMSE_lesion", "nRMSE_c_lesion", "nRMSE_full_brain"],
                 ["GM", "WM", "lesion", "contra-lesion", "brain"], "tracer nRMSE (lower is better)"),
                (["AUC_CBF", "AUC_CBV", "AUC_MTT"], ["CBF", "CBV", "MTT"], "lesion ROC AUC (higher is better)"))):
            ax = axes[i, j]
            w = 0.8 / len(tags)
            for k, (tag, tl) in enumerate(tags):
                if tag not in s:
                    continue
                x = np.arange(len(keys)) + (k - (len(tags) - 1) / 2) * w
                vals = [s[tag][kk] for kk in keys]
                ax.bar(x, vals, w * 0.92, color=C[k], label=tl)
                for xx, vv in zip(x, vals):
                    ax.text(xx, vv, f"{vv:.3f}", ha="center", va="bottom", fontsize=5.8, rotation=90, color=INK)
            ax.set_xticks(range(len(keys)), names)
            ax.set_title(f"{LABEL.get(cfg, cfg)}: {title}", loc="left", fontsize=9, color=INK)
            if j == 1:
                lo = min(s[t][kk] for t, _ in tags if t in s for kk in keys)
                ax.set_ylim(max(0, lo - 0.08), 1.0)
            else:
                ax.set_ylim(0, 1.25 * max(s[t][kk] for t, _ in tags if t in s for kk in keys))
        axes[i, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "npde_metrics.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=str(ROOT / "outputs"))
    p.add_argument("--data-root", default=str(ROOT.parent / "HemoPIC" / "data" / "ISLES2017_Training"))
    p.add_argument("--configs", nargs="*", default=["full", "full_aif"])
    p.add_argument("--only", nargs="*", help="residuals examples fields metrics")
    a = p.parse_args()
    out, data_root = Path(a.out_root), Path(a.data_root)
    FIG.mkdir(parents=True, exist_ok=True)
    which = set(a.only or ["residuals", "examples", "fields", "metrics"])
    if "residuals" in which:
        fig_residuals(out, data_root, a.configs)
    if "examples" in which:
        for c in a.configs:
            fig_examples(out, data_root, c)
    if "fields" in which:
        fig_fields(out, data_root, a.configs)
    if "metrics" in which:
        fig_metrics(out, a.configs)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
