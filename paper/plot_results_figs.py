"""Result figures of the paper from the per-patient export
(scripts/fig_per_patient_export.py).

python paper/plot_results_figs.py fig_per_patient.json figures

hypo_error_ci     median |hypoperfusion error| against the vendor Tmax > 6 s volume by scan length,
                  95% bootstrap CI (patients resampled; ISLES'18 by patient, its slabs together)
decision_fig      (a) signed log1p error of the hypoperfused volume at 30 s, per patient
                  (b) DEFUSE 3 decisions that flip from the full to the truncated scan, ISLES'24,
                      Wilson 95% CI
                  (c) share of the 90 multiverse analyses in which each patient is eligible
Prints the bootstrap CIs at 30 s (for the text).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

M = [("osvd", "oSVD", "#2f5fb3", "-"), ("ssvd", "sSVD", "#3a9a4a", "-"),
     ("gamma_ssvd", "gamma + sSVD", "#e0a020", "-"), ("learned_ssvd", "learned + sSVD", "#7d4cc0", "-"),
     ("duofusion", "physics twin", "#d63a3a", "-"), ("duofusion_w05", "DuoFusion", "#111111", "-")]
HYB = ("hybrid", "sSVD core + twin", "#8c6d31", "--")
XS = ["25", "30", "35", "40", "45", "full"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.titlesize": 7.5, "pdf.fonttype": 42,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2})
RNG = np.random.default_rng(0)


def boot_median(err, groups, B=2000):
    """Median and 95% CI, resampling the groups (patients)."""
    g = np.asarray(groups)
    ug = np.unique(g)
    idx = {u: np.flatnonzero(g == u) for u in ug}
    meds = []
    for _ in range(B):
        pick = RNG.choice(ug, ug.size)
        meds.append(np.median(err[np.concatenate([idx[u] for u in pick])]))
    return float(np.median(err)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def defuse3(core, union):
    return bool(core < 70 and (union / core >= 1.8 if core > 0 else True) and union - core >= 15)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def hypo_error_ci(d, out):
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.05))
    ci30 = {}
    for ax, yy, title in zip(axs, ("24", "18"), ("ISLES'24 (in-domain)", "ISLES'18 (external)")):
        P, V = d["data"][yy]["patients"], d["data"][yy]["volumes"]
        for key, lab, col, ls in M:
            xs, med, lo, hi = [], [], [], []
            for i, L in enumerate(XS):
                if L not in V.get(key, {}):
                    continue
                ns = list(V[key][L])
                err = np.array([abs(V[key][L][n][1] - P[n]["hypo_ref"]) for n in ns])
                m, a, b = boot_median(err, [P[n]["group"] for n in ns])
                xs.append(i)
                med.append(m)
                lo.append(a)
                hi.append(b)
                if L == "30":
                    ci30[(yy, key)] = (m, a, b)
            xs = np.array(xs)
            seg = xs < 5
            ax.plot(xs[seg], np.array(med)[seg], ls, color=col, label=lab, lw=1.6 if key == "duofusion_w05" else 1.1,
                    marker="o", ms=2.2)
            ax.fill_between(xs[seg], np.array(lo)[seg], np.array(hi)[seg], color=col, alpha=0.12, lw=0)
            if xs[-1] == 5:
                ax.errorbar(5 + 0.07 * (M.index((key, lab, col, ls)) - 2.5), med[-1],
                            yerr=[[med[-1] - lo[-1]], [hi[-1] - med[-1]]], fmt="o", color=col, ms=2.2, lw=0.8, capsize=1.2)
        ax.set_yscale("log")
        ticks = [1, 2, 3, 5, 10, 20, 30, 50, 100]
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(range(6))
        ax.set_xticklabels(["25", "30", "35", "40", "45", "full"])
        ax.axvline(4.5, color="#bbb", lw=0.6, ls=":")
        ax.set_xlabel("scan length (s)")
        ax.set_title(title, pad=2)
        clean(ax)
    axs[0].set_ylabel("|hypoperfusion error| (ml)")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="upper center", ncol=6, handlelength=1.3, columnspacing=0.9,
               borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.9))
    fig.savefig(out / "hypo_error_ci.pdf")
    fig.savefig(out / "hypo_error_ci.png", dpi=300)
    return ci30


def decision_fig(d, out):
    fig = plt.figure(figsize=(4.8, 1.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.9], wspace=0.45)
    # (a) signed error at 30 s, both data sets
    ax = fig.add_subplot(gs[0])
    keys = [m for m in M]
    for j, (yy, off, fc) in enumerate((("24", -0.18, None), ("18", 0.18, "white"))):
        P, V = d["data"][yy]["patients"], d["data"][yy]["volumes"]
        data = []
        for key, *_ in keys:
            ns = list(V[key]["30"])
            data.append(np.array([np.log1p(V[key]["30"][n][1]) - np.log1p(P[n]["hypo_ref"]) for n in ns]))
        bp = ax.boxplot(data, positions=np.arange(len(keys)) + off, widths=0.32, showfliers=False, patch_artist=True,
                        medianprops={"color": "k", "lw": 0.9}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6},
                        boxprops={"lw": 0.6})
        for patch, (_, _, col, _) in zip(bp["boxes"], keys):
            patch.set_facecolor(col if yy == "24" else "white")
            patch.set_edgecolor(col)
            patch.set_alpha(0.55 if yy == "24" else 1.0)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["oSVD", "sSVD", "gamma", "learned", "twin", "DuoFus."], rotation=40, ha="right")
    ax.set_ylabel("log(1+V) − log(1+V$_{\\rm ref}$)")
    ax.set_title("(a) hypoperfusion bias, 30 s", pad=2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#999", alpha=0.55, edgecolor="#555", label="ISLES'24"),
                       Patch(facecolor="white", edgecolor="#555", label="ISLES'18")],
              frameon=False, loc="lower right", handlelength=1.0, borderaxespad=0.0)
    clean(ax)
    # (b) DEFUSE 3 flips vs length, ISLES'24
    ax = fig.add_subplot(gs[1])
    P, V = d["data"]["24"]["patients"], d["data"]["24"]["volumes"]
    for key, lab, col, ls in [*M, HYB]:
        xs, ps, los, his = [], [], [], []
        for i, L in enumerate(XS[:5]):
            if L not in V.get(key, {}):
                continue
            ns = [n for n in V[key][L] if n in V[key]["full"]]
            fl = [defuse3(V[key][L][n][0], V[key][L][n][2]) != defuse3(V[key]["full"][n][0], V[key]["full"][n][2])
                  for n in ns]
            p, a, b = wilson(sum(fl), len(fl))
            xs.append(int(L))
            ps.append(100 * p)
            los.append(100 * a)
            his.append(100 * b)
        ax.plot(xs, ps, ls, color=col, marker="o", ms=2.0, lw=1.6 if key == "duofusion_w05" else 1.0)
        ax.fill_between(xs, los, his, color=col, alpha=0.10, lw=0)
    ax.set_xticks([25, 30, 35, 40, 45])
    ax.set_xlabel("scan length (s)")
    ax.set_ylabel("DEFUSE 3 flips (%)")
    ax.set_title("(b) decision flips, ISLES'24", pad=2)
    ax.plot([], [], HYB[3], color=HYB[2], label=HYB[1])
    ax.legend(frameon=False, loc="upper right", handlelength=1.4, borderaxespad=0.0)
    clean(ax)
    # (c) multiverse fragility
    ax = fig.add_subplot(gs[2])
    for col_name, lab, col in (("defuse3_share", "DEFUSE 3", "#2f5fb3"), ("dawn_share", "DAWN", "#d63a3a")):
        s = np.sort([float(p[col_name]) for p in P.values() if p.get(col_name) not in (None, "", "None")])
        ax.plot(np.arange(s.size) / (s.size - 1) * 100, 100 * s, color=col, lw=1.2, label=lab)
    ax.axhspan(10, 90, color="#f2c94c", alpha=0.25, lw=0)
    ax.text(97, 50, "unstable", fontsize=6, va="center", ha="right", color="#8a6d00")
    ax.set_xlabel("patients (sorted, %)")
    ax.set_ylabel("eligible in analyses (%)")
    ax.set_title("(c) fragility", pad=2)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, borderaxespad=0.0, bbox_to_anchor=(0.0, 0.88))
    clean(ax)
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.27, top=0.9)
    fig.savefig(out / "decision_fig.pdf")
    fig.savefig(out / "decision_fig.png", dpi=300)


def main():
    d = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    ci30 = hypo_error_ci(d, out)
    decision_fig(d, out)
    for (yy, key), (m, a, b) in sorted(ci30.items()):
        print(f"ISLES'{yy} {key:14s} 30 s: {m:.1f} [{a:.1f}, {b:.1f}]")


if __name__ == "__main__":
    main()
