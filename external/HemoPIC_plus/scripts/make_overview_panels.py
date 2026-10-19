"""The small plots of the overview figure (paper Fig. 1), from the real fits of one patient.

python scripts/make_overview_panels.py configs/full_aif.yaml --data-root <ISLES2017_Training> [--patient 22] [--out <dir>]

Each panel is drawn at the printed size of its slot in the overview figure (pixels of the
3600-px figure at 541 px per printed inch), so --fs is the printed font size; both axes with
tick labels, axis labels and a legend, as in the original panels. Panels:
concentration of grey-matter voxels, observed ROI curves, automatic AIF with the VOF
partial-volume correction, delayed and dispersed ROI input, Windkessel outflow, curve fit
with the 95% posterior band, and the two tubes of the neural PDE. Needs the fits of <name>
and <name>__npde (with uq/posterior.npz).
Writes <out>/panel_*.png (600 dpi, transparent), panel_*.pdf and panels.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib.ticker  # noqa: E402

from make_figures import C, INK2  # noqa: E402

from hemopic_plus.aif import select_aif  # noqa: E402
from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, load_nifti, squeeze_ctc, subject_id  # noqa: E402
from hemopic_plus.npde import transport as TP  # noqa: E402
from hemopic_plus.npde.uqbatch import load_npde_subject  # noqa: E402
from hemopic_plus.signal import choose_peak_threshold, compute_peak_map, smooth_curve_gaussian  # noqa: E402
from hemopic_plus.uq.torch_model import DTYPE, interp_shift, simulate  # noqa: E402

PX_PER_IN = 541.0
SLOTS = {"concentration": (520, 160), "roi_curves": (520, 180), "aif": (520, 185), "delay": (400, 160),
         "windkessel": (400, 170), "curve_fit": (800, 155), "tubes": (320, 430)}
RED = "#c8372d"


class Panel:
    """One small plot with both axes, tick labels, axis labels and a legend, at the printed size of its slot."""

    def __init__(self, name, fs, ylabel, xlabel="time (s)"):
        w, h = SLOTS[name]
        self.name, self.fs = name, fs
        self.fig, self.ax = plt.subplots(figsize=(w / PX_PER_IN, h / PX_PER_IN))
        ax = self.ax
        ax.set_xlabel(xlabel, fontsize=fs, labelpad=0.5)
        ax.set_ylabel(ylabel, fontsize=fs, labelpad=0.5)
        ax.tick_params(labelsize=fs, length=1.5, width=0.4, pad=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.4)
        ax.grid(True, lw=0.3, color="#e4e3df")
        ax.set_axisbelow(True)
        ax.patch.set_alpha(0)

    def legend(self, loc="upper right", outside=False, room=0.0, head=0.0, **kw):
        """room / head: extra fraction of the time / value range kept free for the legend."""
        if room:
            x0, x1 = self.ax.get_xlim()
            self.ax.set_xlim(x0, x1 + room * (x1 - x0))
        if head:
            y0, y1 = self.ax.get_ylim()
            self.ax.set_ylim(y0, y1 + head * (y1 - y0))
        if outside:
            kw.update(bbox_to_anchor=(1.0, 1.05))
            loc = "upper left"
        kw.setdefault("handlelength", 1.3)
        kw.setdefault("labelspacing", 0.15)
        kw.setdefault("borderaxespad", 0.1)
        self.ax.legend(fontsize=self.fs, frameon=False, loc=loc, handletextpad=0.4, **kw)

    def collisions(self, pad=1.5):
        """Text that touches a curve, the band, or other text (in display pixels)."""
        r = self.fig.canvas.get_renderer()
        ax = self.ax
        boxes = []
        if ax.get_legend() is not None:
            boxes.append(("legend", ax.get_legend().get_window_extent(r)))
        boxes += [(t.get_text(), t.get_window_extent(r)) for t in ax.texts]
        ticks = [lab.get_window_extent(r) for lab in ax.get_xticklabels() + ax.get_yticklabels() if lab.get_text()]
        pts = []
        for line in ax.lines:
            xy = line.get_xydata()
            if len(xy) > 1:
                u = np.linspace(0, len(xy) - 1, 20 * len(xy))
                xy = np.column_stack([np.interp(u, np.arange(len(xy)), xy[:, 0]), np.interp(u, np.arange(len(xy)), xy[:, 1])])
            pts.append(ax.transData.transform(xy))
        for coll in ax.collections:
            for path in coll.get_paths():
                pts.append(ax.transData.transform(path.vertices))
        pts = np.vstack(pts) if pts else np.zeros((0, 2))
        hits = []
        for name, b in boxes:
            inside = (pts[:, 0] > b.x0 - pad) & (pts[:, 0] < b.x1 + pad) & (pts[:, 1] > b.y0 - pad) & (pts[:, 1] < b.y1 + pad)
            if inside.any():
                hits.append(name + " / curve")
            if any(b.overlaps(tb) for tb in ticks):
                hits.append(name + " / tick label")
        frame = ax.get_window_extent(r)
        for name, b in boxes:
            if b.y0 < frame.y0 + pad or b.x0 < frame.x0 + pad:
                hits.append(name + " / axis")
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][1].overlaps(boxes[j][1]):
                    hits.append(boxes[i][0] + " / " + boxes[j][0])
        return hits

    def save(self, out, xticks=(0, 20, 40)):
        self.ax.set_xticks(list(xticks))
        self.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        self.fig.tight_layout(pad=0.15)
        for _ in range(60):
            hits = self.collisions()
            if not hits:
                break
            y0, y1 = self.ax.get_ylim()
            self.ax.set_ylim(y0, y1 + 0.05 * (y1 - y0))
            self.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            self.fig.tight_layout(pad=0.15)
        else:
            raise ValueError(f"{self.name}: text still touches: {hits}")
        tb = self.fig.get_tightbbox(self.fig.canvas.get_renderer())
        w, h = self.fig.get_size_inches()
        x0, y0, x1, y1 = tb.x0, tb.y0, tb.x1, tb.y1
        if x0 < -0.005 or y0 < -0.005 or x1 > w + 0.005 or y1 > h + 0.005:
            raise ValueError(f"{self.name}: content does not fit the slot ({tb})")
        for ext in ("png", "pdf"):
            self.fig.savefig(out / f"panel_{self.name}.{ext}", dpi=600, transparent=True)
        plt.close(self.fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--data-root")
    p.add_argument("--out-root")
    p.add_argument("--patient", type=int, default=22)
    p.add_argument("--fs", type=float, default=3.0, help="printed font size (pt)")
    p.add_argument("--delay-roi", type=int, default=10)
    p.add_argument("--roi", type=int, default=3, help="ROI of the flow, fit and two-tube panels")
    p.add_argument("--out", type=Path, default=Path("docs/figures/overview_panels"))
    a = p.parse_args()
    cfg = load_config(a.config, {k: v for k, v in (("data_root", a.data_root), ("out_root", a.out_root)) if v})
    out_root, data_root = cfg.resolve("out_root"), cfg.resolve("data_root")
    sub = subject_id(a.patient)
    fit_dir, npde_dir, base = out_root / cfg.name / sub, out_root / f"{cfg.name}__npde" / sub, data_root / sub
    a.out.mkdir(parents=True, exist_ok=True)
    fs, lw = a.fs, 0.7
    plt.rcParams.update({"font.size": fs, "lines.linewidth": lw})

    b = load_npde_subject(fit_dir, npde_dir)
    t = b.t.numpy() - float(b.t[0])
    z_tank, z_npde = np.load(fit_dir / "fit_curves.npz"), np.load(npde_dir / "fit_curves.npz")
    zi, zn = list(z_tank["roi_ids"]), list(z_npde["roi_ids"])
    rid = [int(r) for r in b.roi_ids]
    k_roi, k_del = rid.index(a.roi), rid.index(a.delay_roi)
    th = b.theta_fit
    shape = b.shape
    tissue = {int(r["roi_id"]): r["tissue"] for r in csv.DictReader(open(fit_dir / "roi_fit_summary.csv", encoding="utf8"))}

    _, raw = load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))
    ctc = squeeze_ctc(raw)
    seg = load_nifti(base / SEG4_REL)[1].astype(int)
    ot = nib.load(str(find_nii(base, "OT"))).get_fdata() > 0
    lab = nib.load(str(fit_dir / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata().astype(int)
    tt = np.arange(ctc.shape[3]) * b.dt

    P = Panel("concentration", fs, "C(t)")
    gm = (seg == 1) & np.all(np.isfinite(ctc), axis=3)
    curves = ctc[gm]
    pk = curves.max(1)
    pick = np.random.default_rng(0).choice(np.where((pk > np.quantile(pk, 0.6)) & (pk < np.quantile(pk, 0.95)))[0], 6, replace=False)
    for c, col in zip(curves[pick], plt.get_cmap("Blues")(np.linspace(0.45, 0.95, 6))):
        P.ax.plot(tt, smooth_curve_gaussian(c, 1.0, cfg.baseline_frames, False), color=col, lw=0.6)
    x0, x1 = P.ax.get_xlim()
    P.ax.set_xlim(x0, x1 + 0.45 * (x1 - x0))
    P.ax.text(0.98, 0.95, "6 grey-matter voxels", transform=P.ax.transAxes, ha="right", va="top", fontsize=fs, color=INK2)
    P.save(a.out)

    P = Panel("roi_curves", fs, "C(t)")
    overlap = {r: (ot & (lab == r)).sum() / max((lab == r).sum(), 1) for r in rid}
    les = max(rid, key=lambda r: overlap[r])
    ok = [r for r in rid if z_tank["success"][zi.index(r)] and z_tank["c_obs"][zi.index(r)].max() > 0.2 * z_tank["c_obs"].max()]
    clean = sorted((r for r in ok if r != les), key=lambda r: overlap[r])
    others = [r for r in clean if tissue[r] == "GM"][:1] + [r for r in clean if tissue[r] == "WM"][:1]
    chosen = [(les, f"ROI {les} (lesion)", C[1])] + [(r, f"ROI {r}", col) for r, col in zip(others, (C[0], C[2]))]
    for r, name, col in chosen:
        P.ax.plot(t, z_tank["c_obs"][zi.index(r)], color=col, label=name)
    P.legend(room=1.1, labelspacing=0.02, handlelength=1.0, borderaxespad=0.0)
    P.save(a.out)

    P = Panel("aif", fs, "$C_a(t)$")
    show = (seg > 0) & np.all(np.isfinite(ctc), axis=3)
    peak3 = compute_peak_map(ctc, cfg.baseline_frames)
    fitm = show & (peak3 > choose_peak_threshold(peak3, show))
    tissue_mean = smooth_curve_gaussian(ctc[fitm].mean(axis=0), cfg.smooth_sigma_frames, cfg.baseline_frames, True)
    res = select_aif(ctc, show, tt, cfg, seed=cfg.seed + a.patient)
    P.ax.plot(tt, res["aif_unscaled"], color=RED, lw=0.6, ls="--", label="AIF raw")
    P.ax.plot(tt, res["aif"], color=RED, lw=1.0, label="AIF (PVE corr.)")
    if res["vof"] is not None:
        P.ax.plot(tt, res["vof"], color=C[0], label="VOF")
    P.legend(room=1.0, labelspacing=0.02, handlelength=1.0, borderaxespad=0.0)
    P.save(a.out)

    P = Panel("delay", fs, "$C_a$")
    with torch.no_grad():
        shifted = TP.disperse_rows(interp_shift(shape, b.t, th[k_del:k_del + 1, 5]), b.dt,
                                   torch.exp(th[k_del:k_del + 1, 6]))[0].numpy()
    s0 = shape.numpy()
    P.ax.plot(t, s0 / s0.max(), color=RED, lw=0.6, ls="--", label="AIF")
    P.ax.plot(t, shifted / shifted.max(), color=RED, lw=1.0, label=f"ROI, +{float(th[k_del, 5]):.1f} s")
    P.legend(room=1.1, head=0.1)
    P.save(a.out)

    P = Panel("windkessel", fs, "q(t)")
    with torch.no_grad():
        q, _, _ = TP.drive(b.cfg, b.t[None, :], b.dt, shape[None, :], th[k_roi:k_roi + 1])
    q = q[0].numpy()
    F = float(torch.exp(th[k_roi, 3]))
    P.ax.plot(t, q, color=C[0], lw=1.0)
    P.ax.axhline(F, color=INK2, lw=0.5, ls=":")
    P.ax.set_ylim(min(0.0, q.min()), F * 1.18)
    P.ax.text(t[2], F * 0.86, "$F$", fontsize=fs, color=INK2, va="top", ha="left")
    P.save(a.out)

    P = Panel("curve_fit", fs, "C(t)")
    zp = np.load(npde_dir / "uq" / "posterior.npz")
    kz = [int(r) for r in zp["roi_ids"]].index(a.roi)
    draws = torch.as_tensor(zp["theta"][kz][::5], dtype=DTYPE)
    with torch.no_grad():
        _, cd = simulate(b.subset([k_roi] * draws.shape[0]), draws, roi_index=list(range(draws.shape[0])))
    lo, hi = np.percentile(cd.numpy(), [2.5, 97.5], axis=0)
    P.ax.fill_between(t, lo, hi, color=C[2], alpha=0.3, lw=0, label="95% band")
    P.ax.plot(t, z_tank["c_fit"][zi.index(a.roi)], color=C[0], ls="--", label="compartment")
    P.ax.plot(t, z_npde["c_fit"][zn.index(a.roi)], color=C[2], lw=1.0, label="neural PDE")
    P.ax.plot(t, z_npde["c_obs"][zn.index(a.roi)], "o", ms=1.1, color=INK2, mew=0, label="measured")
    P.legend(ncol=2, columnspacing=0.6, handlelength=1.0, room=1.05)
    P.save(a.out, xticks=(0, 10, 20, 30, 40, 50))

    P = Panel("tubes", fs, "$\\hat{C}(t)$")
    nu, phi = b.nu[k_roi:k_roi + 1], b.phi[k_roi:k_roi + 1]
    with torch.no_grad():
        sh = TP.disperse_rows(interp_shift(shape, b.t, th[k_roi:k_roi + 1, 5]), b.dt, torch.exp(th[k_roi:k_roi + 1, 6]))
        _, kdt, u = TP.drive(b.cfg, b.t[None, :], b.dt, sh, th[k_roi:k_roi + 1])
        parts = [TP.tissue(kdt, u, th[k_roi:k_roi + 1, 7], b.n_cells, (nu[:, m:m + 1], phi[:, m:m + 1]))[0].numpy()
                 for m in (0, 1)]
    m_fast = 0 if float(phi[0, 0] / nu[0, 0]) >= float(phi[0, 1] / nu[0, 1]) else 1
    fast, slow = parts[m_fast], parts[1 - m_fast]
    P.ax.plot(t, fast, color=C[0], label="fast tube")
    P.ax.plot(t, slow, color=C[1], label="slow tube")
    P.ax.plot(t, fast + slow, color=C[2], lw=1.2, label="$\\hat{C}(t)$")
    P.legend(head=0.35)
    P.save(a.out)

    info = {"patient": a.patient, "delay_roi": a.delay_roi, "roi": a.roi, "lesion_roi": int(les),
            "roi_curves": [int(r) for r, _, _ in chosen], "fast_tube_nu_phi": [float(nu[0, m_fast]), float(phi[0, m_fast])],
            "pve_scale": float(res["pve_scale"]), "printed_font_pt": fs, "slot_px_at_541_px_per_in": SLOTS}
    (a.out / "panels.json").write_text(json.dumps(info, indent=1), encoding="utf8")
    print(info)


if __name__ == "__main__":
    main()
