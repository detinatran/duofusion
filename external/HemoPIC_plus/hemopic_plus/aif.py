"""[C6] Automatic arterial input function (AIF) and venous output function (VOF).

HemoPIC drives every ROI with C_art(t) = mean curve of all high-peak *tissue*
voxels. That curve is already delayed and dispersed by the capillary bed, so
transit times measured against it are relative to "average tissue" and the
absolute MTT scale is lost (CVT slope ~0.27 once units are fixed, see
docs/RESULTS.md). Here a true arterial curve is selected automatically,
following the clustering approach of Mouridsen et al. (MRM 2006):

  1. candidates: brain voxels in the top `aif_auc_top` fraction of area under
     the baseline-subtracted curve, minus the roughest `aif_rough_drop`;
  2. k-means (k=5) on AUC-normalised curve shapes, keep the cluster with the
     smallest first moment (earliest, narrowest bolus); repeat once;
  3. AIF = mean raw curve of the surviving voxels (at most `aif_max_voxels`,
     highest peak first).

Arterial voxels at DSC resolution suffer partial volume, which scales the AIF
down. The VOF (superior sagittal sinus: large, late, high-AUC voxels) is less
affected, so the AIF is rescaled by AUC(VOF) / AUC(AIF) when that ratio
exceeds 1 (`aif_pve: vof`).

Uncertainty ensemble (scripts/run_aif_ensemble.py): `aif_member > 0` bootstraps
the selected arterial voxels and multiplies the PVE scale by a log-normal factor
(sd `aif_pve_jitter`), both seeded by the member; `aif_cluster_rank: 1` keeps
the second earliest shape cluster in the first clustering round. The defaults
(member 0, rank 0) give the AIF unchanged.
"""
import numpy as np

from .signal import kmeans_numpy, positive_excursion, smooth_curve_gaussian


def _first_moment(d, t):
    s = np.sum(d, axis=1)
    return np.sum(d * t[None, :], axis=1) / np.maximum(s, 1e-12)


def _shape_cluster_earliest(curves, t, seed, k=5, rank=0):
    """Cluster AUC-normalised shapes, return mask of the earliest-moment cluster
    (rank 1: the second earliest, used by the AIF uncertainty ensemble)."""
    if curves.shape[0] <= k:
        return np.ones(curves.shape[0], bool)
    shapes = curves / np.maximum(np.sum(curves, axis=1, keepdims=True), 1e-12)
    lab, _ = kmeans_numpy(shapes.astype(np.float32), k=k, seed=seed, max_iter=50)
    m1 = _first_moment(curves, t)
    order = sorted((j for j in range(k) if np.any(lab == j)), key=lambda j: float(np.median(m1[lab == j])))
    return lab == order[min(int(rank), len(order) - 1)]


def _bolus_quality(raw, d, bf, tail_max):
    """Voxels whose curve looks like a single bolus passage: peak after the
    baseline, baseline SNR >= 10 and a tail that falls below tail_max * peak
    (rejects noise spikes and drifting 'plateau' voxels)."""
    t_len = d.shape[1]
    peak = np.max(d, axis=1)
    ttp = np.argmax(d, axis=1)
    noise = np.std(raw[:, : max(int(bf), 2)], axis=1)
    tail = np.mean(d[:, int(0.8 * t_len):], axis=1)
    return (ttp > bf) & (ttp < t_len - 3) & (peak >= 10.0 * np.maximum(noise, 1e-12)) & (tail <= tail_max * peak) & (peak > 0)


def select_aif(ctc, mask3, t, cfg, seed=0):
    """Return dict with the AIF (absolute units of the CTC), VOF and diagnostics,
    or None when no plausible arterial cluster is found (caller falls back to
    the tissue-mean curve)."""
    raw = ctc[mask3].astype(np.float64)
    coords = np.argwhere(mask3)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    bf = int(cfg.baseline_frames)
    d = positive_excursion(raw, bf)
    auc = np.sum(d, axis=1)
    peak = np.max(d, axis=1)
    good = _bolus_quality(raw, d, bf, tail_max=0.5)
    if np.sum(good) < 50:
        return None

    cand = good & (auc >= np.quantile(auc[good], 1.0 - float(cfg.aif_auc_top)))
    rough = np.sum(np.diff(d, n=2, axis=1) ** 2, axis=1) / np.maximum(peak, 1e-12) ** 2
    cand &= rough <= np.quantile(rough[cand], 1.0 - float(cfg.aif_rough_drop))
    idx = np.where(cand)[0]
    if idx.size < 10:
        return None

    for rnd in range(2):
        keep = _shape_cluster_earliest(d[idx], t, seed + rnd, rank=int(cfg.aif_cluster_rank) if rnd == 0 else 0)
        if np.sum(keep) < 3:
            break
        idx = idx[keep]
    idx = idx[np.argsort(-peak[idx])][: int(cfg.aif_max_voxels)]
    rng = np.random.default_rng(1000003 * int(cfg.aif_member) + int(seed)) if int(cfg.aif_member) > 0 else None
    sel = rng.choice(idx, idx.size, replace=True) if rng is not None else idx

    aif = smooth_curve_gaussian(raw[sel].mean(axis=0), cfg.aif_smooth_sigma, cfg.baseline_frames, True)
    aif_d = positive_excursion(aif[None, :], cfg.baseline_frames)[0]
    aif_ttp = int(np.argmax(aif_d))

    tissue_d = positive_excursion(raw.mean(axis=0)[None, :], bf)[0]
    aif_m1 = float(_first_moment(aif_d[None, :], t)[0])
    tissue_m1 = float(_first_moment(tissue_d[None, :], t)[0])
    dt = float(t[1] - t[0])
    half = aif_d >= 0.5 * aif_d.max()
    fwhm = float(np.sum(half) * dt)
    tissue_fwhm = float(np.sum(tissue_d >= 0.5 * tissue_d.max()) * dt)
    if not (aif_m1 < tissue_m1 and fwhm <= tissue_fwhm and aif_d.max() > 2.0 * tissue_d.max()):
        return None

    ttp = np.argmax(d, axis=1)
    vgood = _bolus_quality(raw, d, bf, tail_max=0.6) & (ttp >= aif_ttp + 2)
    vof, scale = None, 1.0
    if np.sum(vgood) >= 3:
        vcand = np.where(vgood & (peak >= np.quantile(peak[vgood], 0.99)))[0]
        vidx = vcand[np.argsort(-peak[vcand])][: int(cfg.aif_max_voxels)]
        vof = smooth_curve_gaussian(raw[vidx].mean(axis=0), cfg.aif_smooth_sigma, bf, True)
        if cfg.aif_pve == "vof":
            ratio = float(np.sum(positive_excursion(vof[None, :], bf)) / max(np.sum(aif_d), 1e-12))
            scale = float(np.clip(ratio, 1.0, float(cfg.aif_pve_max)))
            if rng is not None:
                scale = float(np.clip(scale * np.exp(rng.normal(0.0, float(cfg.aif_pve_jitter))), 1.0, float(cfg.aif_pve_max)))
    aif_scaled = aif * scale

    return {
        "aif": aif_scaled,
        "aif_unscaled": aif,
        "vof": vof,
        "pve_scale": scale,
        "n_voxels": int(idx.size),
        "voxels": coords[idx],
        "ttp_s": float(t[aif_ttp]),
        "fwhm_s": fwhm,
        "tissue_fwhm_s": tissue_fwhm,
        "first_moment_s": aif_m1,
        "tissue_first_moment_s": tissue_m1,
    }
