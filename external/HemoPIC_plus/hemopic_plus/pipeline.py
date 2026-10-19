"""Per-subject pipeline: partition -> ROI fits -> CBF/CBV/MTT maps.

Mirrors HemoPIC's run_core (MIT License, LDR Group @ JHU) operation by
operation; the [C1]-[C5] options only change behaviour when enabled. Outputs
keep HemoPIC's file layout so the original Eval/ scripts also run on them, and
add fit_curves.npz (fitted ROI curves for any model variant).
"""
import csv
import json
from pathlib import Path

import numpy as np

from .aif import select_aif
from .fit import fit_roi_curve
from .io import (SEG4_REL, ensure_clean_dir, find_nii, load_nifti, resolve_dt,
                 save_nifti_like, squeeze_ctc, subject_id)
from .models import tracer_update
from .osvd import osvd_maps
from .signal import (choose_peak_threshold, compute_features_for_kmeans, compute_peak_map,
                     fill_nearest, kmeans_numpy, merge_small_clusters, positive_excursion,
                     robust_median_positive, smooth_curve_gaussian, voxel_auc_from_ctc,
                     voxel_proxy_from_ctc)

CSV_COLS = ["roi_id", "tissue", "n_vox_all", "n_vox_fit", "F_prior", "V_prior", "Cin_prior",
            "R_fit", "C_wk_fit", "tau_fit", "F_fit", "V_fit", "Cin_bar_fit", "loss", "success", "message",
            "delay_fit", "dispersion_fit", "objective", "n_starts"]


def _build_partition(cfg, ctc, seg4, show_mask3, dt, seed, k_gm, k_wm):
    def subclusters(tissue_id, k, offset):
        sel3 = (seg4 == int(tissue_id)) & show_mask3
        coords = np.argwhere(sel3)
        n = int(coords.shape[0])
        out = np.zeros(seg4.shape, dtype=np.int32)
        if n < 800:
            ids = []
            if n > 0:
                out[coords[:, 0], coords[:, 1], coords[:, 2]] = offset + 1
                ids.append(offset + 1)
            return out, ids
        feats = compute_features_for_kmeans(ctc[sel3].astype(np.float32), coords, dt=dt, baseline_frames=cfg.baseline_frames)
        feats[:, 3:6] = feats[:, 3:6] * float(cfg.coord_weight)
        rs = np.random.RandomState(int(seed) + int(tissue_id) + 7)
        feats_fit = feats[rs.choice(n, size=int(cfg.sample_max), replace=False).astype(np.int64)] if n > int(cfg.sample_max) else feats
        _, cent = kmeans_numpy(feats_fit, k=int(k), seed=int(seed) + int(tissue_id) + 11, max_iter=int(cfg.kmeans_max_iter))
        best_dist = np.full(n, np.inf, dtype=np.float32)
        best_lab = np.zeros(n, dtype=np.int32)
        for j in range(int(cent.shape[0])):
            diff = np.subtract(feats, cent[j])
            dist = np.sum(diff * diff, axis=1).astype(np.float32)
            take = dist < best_dist
            best_dist[take] = dist[take]
            best_lab[take] = j
        lab2, _ = merge_small_clusters(best_lab, cent, min_vox=int(cfg.min_cluster_vox))
        ids = []
        for j in np.unique(lab2).astype(np.int32).tolist():
            pts = coords[lab2 == j]
            out[pts[:, 0], pts[:, 1], pts[:, 2]] = offset + j + 1
            ids.append(offset + j + 1)
        return out, ids

    gm3, gm_ids = subclusters(1, k_gm, 0)
    off2 = max(gm_ids) if gm_ids else 0
    wm3, wm_ids = subclusters(2, k_wm, off2)
    off3 = max(wm_ids) if wm_ids else off2
    csf_id, bs_id = off3 + 1, off3 + 2
    sub3 = np.zeros(seg4.shape, dtype=np.int32)
    sub3[gm3 > 0] = gm3[gm3 > 0]
    sub3[wm3 > 0] = wm3[wm3 > 0]
    sub3[(seg4 == 3) & show_mask3] = csf_id
    sub3[(seg4 == 4) & show_mask3] = bs_id
    sub3 = fill_nearest(sub3, show_mask3, show_mask3 & (sub3 > 0))
    return sub3, gm_ids, wm_ids, csf_id, bs_id


def _indicator_dilution(cfg, rois, cart_smooth, dt, absolute=False):
    """CBV by indicator dilution (area ratio to the arterial curve) and CBF by
    the maximum-slope method, per ROI. Scale: absolute units with a true AIF
    ([C6]), otherwise one factor per subject mapping normal-appearing white
    matter to ref_wm_cbf / ref_wm_cbv. Sets r["cbf_id"], r["cbv_id"] and returns
    (k_cbf, k_cbv, arterial AUC)."""
    art = positive_excursion(cart_smooth[None, :], cfg.baseline_frames)[0]
    art_auc = float(max(np.sum(art) * dt, 1e-12))
    art_peak = float(max(np.max(art), 1e-12))
    for r in rois:
        d = positive_excursion(r["c_smooth"][None, :], cfg.baseline_frames)[0]
        r["cbv_raw"] = float(np.sum(d) * dt / art_auc)
        r["cbf_raw"] = float(60.0 * max(np.max(np.diff(d)) / dt, 0.0) / art_peak)

    def calib(key, ref):
        wm = [(r[key], r["n_all"]) for r in rois if r["tissue"] == "WM" and r[key] > 0]
        pool = wm if wm else [(r[key], r["n_all"]) for r in rois if r[key] > 0]
        if not pool:
            return 1.0
        vals = np.array([p[0] for p in pool])
        wts = np.array([p[1] for p in pool], dtype=np.float64)
        order = np.argsort(vals)
        cw = np.cumsum(wts[order])
        med = float(vals[order][np.searchsorted(cw, 0.5 * cw[-1])])
        return float(ref) / max(med, 1e-12)

    if absolute:
        kf = kv = 100.0 * float(cfg.k_h) / float(cfg.rho_brain)
    else:
        kf, kv = calib("cbf_raw", cfg.ref_wm_cbf), calib("cbv_raw", cfg.ref_wm_cbv)
    for r in rois:
        r["cbf_id"] = float(max(kf * r["cbf_raw"], cfg.eps_small))
        r["cbv_id"] = float(max(kv * r["cbv_raw"], cfg.eps_small))
    return kf, kv, art_auc


def _tracer_priors(cfg, rois, cart_smooth, dt, absolute=False):
    """[C3] Reference-free priors from indicator dilution / maximum slope; no
    silver-standard map is read."""
    kf, kv, art_auc = _indicator_dilution(cfg, rois, cart_smooth, dt, absolute)
    for r in rois:
        r["f_prior"] = r["cbf_id"]
        r["v_prior"] = r["cbv_id"]
    return kf, kv, art_auc


def run_subject(cfg, patient_num: int, out_root: Path):
    patient_num = int(patient_num)
    base = cfg.resolve("data_root") / subject_id(patient_num)
    if not base.exists():
        raise FileNotFoundError("Missing patient folder " + str(base))
    paths = {k: find_nii(base, k) for k in ("MR_rCBF", "MR_rCBV", "MR_MTT", "CTC_from_MR_4DPWI")}
    if any(v is None for v in paths.values()):
        raise FileNotFoundError("Missing one of " + ", ".join(paths))
    seg_path = base / SEG4_REL
    if not seg_path.exists():
        raise FileNotFoundError("Missing seg4 file " + str(seg_path))

    cbf_img, cbf3 = load_nifti(paths["MR_rCBF"])
    _, cbv3 = load_nifti(paths["MR_rCBV"])
    _, mtt3 = load_nifti(paths["MR_MTT"])
    ctc_img, ctc_raw = load_nifti(paths["CTC_from_MR_4DPWI"])
    ctc = squeeze_ctc(ctc_raw)
    _, seg4 = load_nifti(seg_path)
    seg4 = seg4.astype(np.int32)
    if tuple(seg4.shape[:3]) != tuple(cbf3.shape[:3]):
        raise RuntimeError("Seg shape mismatch perfusion shape")

    t_len = int(ctc.shape[3])
    dt, dt_src = resolve_dt(cfg, ctc_img, patient_num)
    t = np.arange(t_len, dtype=np.float64) * float(dt)

    fit_dir = Path(out_root) / subject_id(patient_num)
    ensure_clean_dir(fit_dir)

    show_mask3 = (seg4 > 0) & np.all(np.isfinite(ctc), axis=3)
    peak3 = compute_peak_map(ctc, baseline_frames=cfg.baseline_frames)
    thr = choose_peak_threshold(peak3, show_mask3)
    fit_mask3 = show_mask3 & (peak3 > float(thr))
    if int(np.sum(fit_mask3)) < 5000:
        fit_mask3 = show_mask3

    cart_smooth = smooth_curve_gaussian(ctc[fit_mask3].mean(axis=0).astype(np.float64), sigma_frames=cfg.smooth_sigma_frames,
                                        baseline_frames=cfg.baseline_frames, clip_nonneg=cfg.clip_nonneg)
    cart_norm = (cart_smooth / float(max(float(np.max(cart_smooth)), 1.0 / 1e12))).astype(np.float64)
    w_curve = cart_norm

    aif_info = None
    if cfg.arterial_input == "auto_aif":
        res = select_aif(ctc, show_mask3, t, cfg, seed=int(cfg.seed) + patient_num)
        if res is not None:
            cart_smooth = res["aif"].astype(np.float64)
            cart_norm = (cart_smooth / float(max(float(np.max(cart_smooth)), 1.0 / 1e12))).astype(np.float64)
            aif_info = {k: (v if not isinstance(v, np.ndarray) else None) for k, v in res.items()
                        if k not in ("aif", "aif_unscaled", "vof", "voxels")}
            aif_info["voxels"] = res["voxels"].tolist()

    sub3, gm_ids, wm_ids, csf_id, bs_id = _build_partition(cfg, ctc, seg4, show_mask3, dt, cfg.seed, cfg.k_gm, cfg.k_wm)
    unlabeled = int(np.sum(show_mask3 & (sub3 == 0)))
    save_nifti_like(cbf_img, sub3.astype(np.int32), fit_dir / "roi_labels_seg4_within_kmeans.nii.gz", dtype=np.int32)

    rois = []
    for roi_id in gm_ids + wm_ids + [csf_id, bs_id]:
        sel_all = sub3 == int(roi_id)
        n_all = int(np.sum(sel_all))
        if n_all < 200:
            continue
        tissue = "GM" if roi_id in gm_ids else "WM" if roi_id in wm_ids else "CSF" if roi_id == csf_id else "BrainStem"
        sel_fit = sel_all & fit_mask3
        if int(np.sum(sel_fit)) < 80:
            sel_fit = sel_all
        c_smooth = smooth_curve_gaussian(ctc[sel_fit].mean(axis=0).astype(np.float64), sigma_frames=cfg.smooth_sigma_frames,
                                         baseline_frames=cfg.baseline_frames, clip_nonneg=cfg.clip_nonneg)
        rois.append({"roi_id": int(roi_id), "tissue": tissue, "sel_all": sel_all, "n_all": n_all, "n_fit": int(np.sum(sel_fit)),
                     "c_smooth": c_smooth, "cin_prior": float(max(float(np.max(c_smooth)), cfg.eps_small))})

    calib = None
    use_abs = cfg.prior_scale == "absolute" and aif_info is not None
    if cfg.prior_source == "reference_maps":
        for r in rois:
            r["f_prior"] = robust_median_positive(cbf3[r["sel_all"]].astype(np.float64), eps=cfg.eps_small)
            r["v_prior"] = robust_median_positive(cbv3[r["sel_all"]].astype(np.float64), eps=cfg.eps_small)
        if cfg.cbv_source != "model_v":
            calib = _indicator_dilution(cfg, rois, cart_smooth, dt, absolute=use_abs)
    else:
        calib = _tracer_priors(cfg, rois, cart_smooth, dt, absolute=use_abs)

    rng = np.random.default_rng(int(cfg.seed) + 1000 * patient_num)
    fits, rows = {}, []
    for r in rois:
        fit = fit_roi_curve(r["c_smooth"], cart_norm, t, dt, r["f_prior"], r["v_prior"], r["cin_prior"], cfg, rng=rng, w_curve=w_curve)
        if not bool(fit["success"]):
            q_out = np.full(t.shape[0], float(r["f_prior"]), dtype=np.float64)
            c_fit = tracer_update(q_out, float(r["cin_prior"]) * cart_norm, float(r["v_prior"]), float(r["f_prior"]), float(dt),
                                  q_floor_ratio=cfg.q_floor_ratio, flow_per_minute=cfg.flow_per_minute)
            fit = {"R": 1.0, "tau": 1.0, "C_wk": 1.0, "V": float(r["v_prior"]), "F": float(r["f_prior"]), "Cin_bar": float(r["cin_prior"]),
                   "delay": 0.0, "dispersion": 0.0, "loss": float(1e9), "objective": float(1e9), "n_starts": fit["n_starts"],
                   "success": False, "message": "fallback_to_priors", "Q_out": q_out, "C_fit": c_fit}
        fits[r["roi_id"]] = fit
        rows.append([r["roi_id"], r["tissue"], r["n_all"], r["n_fit"], float(r["f_prior"]), float(r["v_prior"]), float(r["cin_prior"]),
                     float(fit["R"]), float(fit["C_wk"]), float(fit["tau"]), float(fit["F"]), float(fit["V"]), float(fit["Cin_bar"]),
                     float(fit["loss"]), bool(fit["success"]), str(fit["message"]), float(fit["delay"]), float(fit["dispersion"]),
                     float(fit["objective"]), int(fit["n_starts"])])

    with open(fit_dir / "roi_fit_summary.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        w.writerows(rows)
    np.savez_compressed(fit_dir / "fit_curves.npz", t=t, dt=float(dt), cart_norm=cart_norm, arterial=cart_smooth, w_curve=w_curve,
                        roi_ids=np.array([r["roi_id"] for r in rois], dtype=np.int32),
                        success=np.array([bool(fits[r["roi_id"]]["success"]) for r in rois]),
                        c_obs=np.stack([r["c_smooth"] for r in rois]) if rois else np.zeros((0, t_len)),
                        c_fit=np.stack([fits[r["roi_id"]]["C_fit"] for r in rois]) if rois else np.zeros((0, t_len)))

    art_d = positive_excursion(cart_smooth[None, :], cfg.baseline_frames)[0]
    art_m1 = float(np.sum(art_d * t) / max(np.sum(art_d), 1e-12))
    shape3 = seg4.shape
    md_cbf = np.full(shape3, np.nan, dtype=np.float64)
    md_cbv = np.full(shape3, np.nan, dtype=np.float64)
    prior_v = {r["roi_id"]: r["v_prior"] for r in rois}
    cbv_id = {r["roi_id"]: r.get("cbv_id") for r in rois}
    for roi_id, fit in fits.items():
        sel_all = sub3 == int(roi_id)
        n_all = int(np.sum(sel_all))
        if n_all < 50:
            continue
        q_out = fit["Q_out"].astype(np.float64)
        tail_len = int(max(3, int(np.floor(float(q_out.size) * float(cfg.tail_frac)))))
        q_tail = q_out[int(max(0, q_out.size - tail_len)):]
        q_tail = q_tail[np.isfinite(q_tail)]
        q_mean = float(max(float(fit["F"]) if q_tail.size < 1 else float(np.mean(q_tail)), cfg.eps_small))
        v_roi = float(max(float(fit["V"]), cfg.eps_small))
        if cfg.cbv_source == "id_regional":
            v_roi = float(cbv_id[int(roi_id)])
        if cfg.cbv_source == "id_voxel":
            md_cbv[sel_all] = calib[1] * voxel_auc_from_ctc(ctc[sel_all].astype(np.float64), dt=dt,
                                                            baseline_frames=cfg.baseline_frames) / calib[2]
        if roi_id in (csf_id, bs_id):
            md_cbf[sel_all] = q_mean
            if cfg.cbv_source != "id_voxel":
                md_cbv[sel_all] = v_roi
            continue

        ctc_roi = ctc[sel_all].astype(np.float64)
        if cfg.cbf_texture == "peak_ttp":
            proxy, _ = voxel_proxy_from_ctc(ctc_roi, dt=dt, baseline_frames=cfg.baseline_frames, eps=cfg.eps_small)
        elif cfg.cbf_texture == "moment":
            d_roi = positive_excursion(ctc_roi, cfg.baseline_frames)
            auc_v = np.sum(d_roi, axis=1)
            m1 = np.sum(d_roi * t[None, :], axis=1) / np.maximum(auc_v, 1e-12)
            mtt_v = np.clip(m1 - art_m1, cfg.moment_mtt_bounds[0], cfg.moment_mtt_bounds[1])
            proxy = np.where(auc_v > 0, auc_v / mtt_v, 0.0)
        else:
            proxy, _, _ = osvd_maps(positive_excursion(ctc_roi, cfg.baseline_frames), art_d, dt)
            proxy = np.where(np.isfinite(proxy), proxy, 0.0)
        proxy_ratio = np.ones(n_all, dtype=np.float64)
        proxy_ok = proxy[np.isfinite(proxy) & (proxy > 0.0)]
        if int(proxy_ok.size) >= 20:
            med = float(max(float(np.median(proxy_ok)), cfg.eps_small))
            proxy_ratio = np.divide(proxy, med)
            proxy_ratio = np.where(np.isfinite(proxy_ratio), proxy_ratio, 1.0)
            if cfg.cbf_texture_clip:
                proxy_ratio = np.clip(proxy_ratio, cfg.proxy_clip_lo, cfg.proxy_clip_hi)
        mean_ratio = float(np.mean(proxy_ratio[np.isfinite(proxy_ratio)])) if int(np.sum(np.isfinite(proxy_ratio))) > 0 else 1.0
        mean_ratio = float(max(mean_ratio, cfg.eps_small))
        cbf_vox = np.divide(q_mean * proxy_ratio, mean_ratio).astype(np.float64)

        if cfg.cbv_texture == "reference_maps":
            src = cbv3[sel_all].astype(np.float64)
            fallback = float(prior_v[int(roi_id)])
        else:
            src = voxel_auc_from_ctc(ctc_roi, dt=dt, baseline_frames=cfg.baseline_frames)
            fallback = 1.0
        pos = src[np.isfinite(src) & (src > 0.0)]
        denom = float(max(float(np.median(pos)) if int(pos.size) >= 20 else fallback, cfg.eps_small))
        cbv_ratio = np.divide(np.where(np.isfinite(src) & (src > 0.0), src, denom), denom)
        cbv_ratio = np.where(np.isfinite(cbv_ratio), cbv_ratio, 1.0)
        cbv_ratio = np.clip(cbv_ratio, cfg.cbv_ratio_clip_lo, cfg.cbv_ratio_clip_hi)
        mean_ratio_v = float(np.mean(cbv_ratio[np.isfinite(cbv_ratio)])) if int(np.sum(np.isfinite(cbv_ratio))) > 0 else 1.0
        mean_ratio_v = float(max(mean_ratio_v, cfg.eps_small))
        md_cbf[sel_all] = cbf_vox
        if cfg.cbv_source != "id_voxel":
            md_cbv[sel_all] = np.divide(v_roi * cbv_ratio, mean_ratio_v).astype(np.float64)

    md_cbf = fill_nearest(md_cbf, show_mask3, show_mask3 & np.isfinite(md_cbf) & (md_cbf > cfg.eps_small))
    md_cbv = fill_nearest(md_cbv, show_mask3, show_mask3 & np.isfinite(md_cbv) & (md_cbv > cfg.eps_small))
    md_cbf = np.where(show_mask3, md_cbf, np.nan)
    md_cbv = np.where(show_mask3, md_cbv, np.nan)

    if cfg.map_bounds == "reference_maps":
        gt_cbf, gt_mtt = cbf3.astype(np.float64), mtt3.astype(np.float64)
        m_cbf = show_mask3 & np.isfinite(gt_cbf) & (gt_cbf > 0.0)
        m_mtt = show_mask3 & np.isfinite(gt_mtt) & (gt_mtt > 0.0)
        cbf_floor, cbf_cap = (float(np.percentile(gt_cbf[m_cbf], 1.0)), float(np.percentile(gt_cbf[m_cbf], 99.5))) if int(np.sum(m_cbf)) > 200 else (1.0, 200.0)
        mtt_floor, mtt_cap = (float(np.percentile(gt_mtt[m_mtt], 1.0)), float(np.percentile(gt_mtt[m_mtt], 99.5))) if int(np.sum(m_mtt)) > 200 else (1.0, 30.0)
    else:
        m_cbf = show_mask3 & np.isfinite(md_cbf) & (md_cbf > 0.0)
        cbf_floor, cbf_cap = (float(np.percentile(md_cbf[m_cbf], 1.0)), float(np.percentile(md_cbf[m_cbf], 99.5))) if int(np.sum(m_cbf)) > 200 else (1.0, 200.0)
        mtt_floor, mtt_cap = float(cfg.mtt_fixed[0]), float(cfg.mtt_fixed[1])
    cbf_floor = float(max(cbf_floor, cfg.eps_small))
    cbf_cap = float(max(cbf_cap, cbf_floor))
    mtt_floor = float(max(mtt_floor, cfg.eps_small))
    mtt_cap = float(max(mtt_cap, mtt_floor))
    den = np.minimum(np.maximum(md_cbf, cbf_floor), cbf_cap)
    md_mtt = np.full_like(md_cbf, np.nan, dtype=np.float64)
    ok3 = show_mask3 & np.isfinite(md_cbv) & np.isfinite(den) & (md_cbv > 0.0) & (den > 0.0)
    md_mtt[ok3] = 60.0 * np.divide(md_cbv[ok3], den[ok3])
    md_mtt = np.where(show_mask3, np.maximum(md_mtt, mtt_floor), np.nan)
    md_mtt = np.where(show_mask3, np.minimum(md_mtt, mtt_cap), np.nan)

    for tag, vol in (("CBF", md_cbf), ("CBV", md_cbv), ("MTT", md_mtt)):
        save_nifti_like(cbf_img, vol, fit_dir / f"HemoPIC_{tag}.nii.gz", dtype=np.float32)

    info = {
        "patient": patient_num, "dt_seconds": float(dt), "k_gm": int(cfg.k_gm), "k_wm": int(cfg.k_wm), "seed": int(cfg.seed),
        "baseline_frames": int(cfg.baseline_frames), "aif_peak_threshold": float(thr), "unlabeled_after_fill": unlabeled,
        "cbf_floor_from_gt_p1": cbf_floor, "cbf_cap_from_gt_p99_5": cbf_cap, "mtt_floor_from_gt_p1": mtt_floor, "mtt_cap_from_gt_p99_5": mtt_cap,
        "dt_source": dt_src, "arterial_input": "auto_aif" if aif_info is not None else "tissue_mean", "aif": aif_info, "n_rois": len(rois), "n_failed": int(sum(not f["success"] for f in fits.values())),
        "prior_calibration": None if calib is None else {"k_cbf": calib[0], "k_cbv": calib[1], "arterial_auc": calib[2]}, "config": cfg.to_dict(),
    }
    (fit_dir / "run_info_core.json").write_text(json.dumps(info, indent=2), encoding="utf8")
    return {"patient": patient_num, "dt": float(dt), "n_rois": len(rois), "n_failed": info["n_failed"],
            "median_loss": float(np.median([f["loss"] for f in fits.values()])) if fits else float("nan")}
