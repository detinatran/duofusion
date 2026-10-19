"""Line-chart panels of the pipeline figure, from the real voxel curves exported by
scripts/fig_curves_export.py.

python paper/plot_pipeline_curves.py voxel_T30.npz figures/pipeline_curves

Writes one PDF, SVG and transparent PNG (600 dpi) per panel, plus _preview.png:
  truncation      measured curve, first L s kept, missing washout shaded
  est_osvd, est_ssvd, est_gamma, est_learned, est_twin
                  truncated curve (dots), hidden washout of the full scan (grey), estimator
                  output (solid on the measured window, dashed beyond L)
  deconv_C, deconv_A, deconv_R   full-scan tissue curve, AIF and oSVD residue (normalised)
  twin_A, twin_C, twin_q         HemoPIC+ input, voxel model curve over the data, Windkessel outflow
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external" / "HemoPIC_plus"))
from hemopic_plus.osvd import _circulant_svd, _oscillation  # noqa: E402


def osvd_residue(c, aif, dt, osc_target=0.095,
                 thresholds=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)):
    """Residue of one curve by the oSVD of hemopic_plus.osvd.osvd_maps (first T samples)."""
    T = c.size
    U, S, Vt = _circulant_svd(aif, dt)
    cpad = np.r_[c, np.zeros(T)]
    for thr in thresholds:
        inv = np.where(S >= thr * S.max(), 1.0 / S, 0.0)
        k = Vt.T @ (inv * (U.T @ cpad))
        if _oscillation(k[None, :])[0] <= osc_target:
            break
    return k[:T]

COL = {"osvd": "#2f5fb3", "ssvd": "#3a9a4a", "gamma": "#e0a020", "learned": "#7d4cc0", "twin": "#d63a3a",
       "grey": "#9a9a9a", "aif": "#d63a3a", "tissue": "#2f5fb3", "vein": "#2a6fd6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.9,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42})


def axes(size=(2.4, 1.35)):
    fig, ax = plt.subplots(figsize=size)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    return fig, ax


def save(fig, out, name, panels):
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", transparent=True)
    fig.savefig(out / f"{name}.png", dpi=600, transparent=True)
    panels.append((name, fig))


def main():
    z = np.load(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    t = z["t_full"]
    T, keep, L = t.size, int(z["keep"]), float(z["seconds"])
    truth, obs = z["truth"], z["obs"]
    tm = t[:keep]
    xmax = t[-1]
    ymax = 1.15 * truth.max()
    panels = []

    # 1. truncation of the measured curve
    fig, ax = axes((2.9, 1.6))
    ax.axvspan(0, L, color="#cfe0f3", alpha=0.7, lw=0)
    ax.axvspan(L, xmax, color="#e3e3e3", alpha=0.8, lw=0)
    ax.plot(tm, obs, "-o", color="#1f3f7a", ms=2.6, lw=1.1)
    ax.plot(t[keep - 1:], truth[keep - 1:], "--o", color=COL["grey"], ms=2.2, lw=0.9, mfc="white")
    ax.text(L / 2, ymax * 0.95, f"measured (first {L:.0f} s)", ha="center", va="top", fontsize=7.5, color="#1f3f7a")
    ax.text((L + xmax) / 2, ymax * 0.95, "missing washout", ha="center", va="top", fontsize=7.5, color="#555")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$C(t)$")
    save(fig, out, "truncation", panels)

    # 2. estimators on the truncated curve (shared y range, wide enough for the gamma overshoot)
    ymax = 1.08 * max(truth.max(), z["gamma"][:T].max(), z["learned"][:T].max(), z["twin_tr_sim"].max())
    est = {"osvd": (z["osvd_fit"], None, "oSVD fit"), "ssvd": (z["ssvd_fit"], None, "sSVD fit"),
           "gamma": (z["gamma"][:T], None, "gamma variate"), "learned": (z["learned"][:T], None, "1D CNN"),
           "twin": (z["twin_tr_sim"], None, "HemoPIC+")}
    for key, (curve, _, _) in est.items():
        fig, ax = axes()
        ax.axvline(L, color="#777", lw=0.7, ls=":")
        ax.plot(t[keep - 1:], truth[keep - 1:], "o", color=COL["grey"], ms=1.8, mfc="white", mew=0.6)
        ax.plot(tm, obs, "o", color=COL[key], ms=2.2, alpha=0.75, mew=0)
        n = curve.size
        if n <= keep:                       # deconvolution: no prediction beyond L
            ax.plot(tm[:n], curve, "-", color=COL[key], lw=1.4)
        else:
            ax.plot(t[:keep], curve[:keep], "-", color=COL[key], lw=1.4)
            ax.plot(t[keep - 1:n], curve[keep - 1:n], "--", color=COL[key], lw=1.4)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("t (s)", labelpad=1)
        ax.set_ylabel(r"$C(t)$", labelpad=2)
        save(fig, out, f"est_{key}", panels)

    # 3. deconvolution thumbnails (full scan)
    small = (1.15, 0.8)
    k = np.clip(osvd_residue(truth, z["aif_full"], float(z["dt"])), 0, None)
    for name, y, col, lab in (("deconv_C", truth, COL["tissue"], r"$C(t)$"),
                              ("deconv_A", z["aif_full"], COL["aif"], r"$A(t)$"),
                              ("deconv_R", k / max(k.max(), 1e-9), "#333333", r"$R(t)$")):
        fig, ax = axes(small)
        ax.plot(t, y, "-", color=col, lw=1.3)
        ax.set_xticks([])
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.1 * np.max(y))
        save(fig, out, name, panels)

    # 4. twin thumbnails (full-scan voxel fit)
    q = z["twin_full_q"]
    for name, ys, cols, lab in (("twin_A", [z["shape_full"]], [COL["aif"]], r"$A(t)$"),
                                ("twin_C", [z["truth_smooth"], z["twin_full_fit"]], [COL["grey"], COL["ssvd"]], r"$C_{\rm model}(t)$"),
                                ("twin_q", [q], [COL["vein"]], r"$q(t)$")):
        fig, ax = axes(small)
        for i, (y, c) in enumerate(zip(ys, cols)):
            if len(ys) > 1 and i == 0:
                ax.plot(t, y, "o", color=c, ms=1.3, mfc="white", mew=0.5)
            else:
                ax.plot(t, y, "-", color=c, lw=1.3)
        ax.set_xticks([])
        ax.set_xlim(0, xmax)
        lo = min(0, min(np.min(y) for y in ys))
        ax.set_ylim(lo, 1.1 * max(np.max(y) for y in ys))
        save(fig, out, name, panels)

    # preview sheet
    import matplotlib.image as mpimg
    names = [n for n, _ in panels]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    pf, pax = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 1.9))
    for a_, n in zip(pax.flat, names + [None] * (rows * cols - len(names))):
        a_.axis("off")
        if n:
            a_.imshow(mpimg.imread(out / f"{n}.png"))
            a_.set_title(n, fontsize=8)
    pf.tight_layout()
    pf.savefig(out / "_preview.png", dpi=110, facecolor="white")
    print("panels:", ", ".join(names), "->", out)


if __name__ == "__main__":
    main()
