"""Curve processing, k-means features and ROI clustering.

Functions marked (HemoPIC) are copied from the original implementation
(MIT License, LDR Group @ JHU) and must stay numerically identical so that the
baseline config reproduces the published outputs.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d


def smooth_curve_gaussian(y, sigma_frames, baseline_frames, clip_nonneg):
    y = np.asarray(y, dtype=np.float64)
    if int(y.size) < 5:
        return y.copy()
    bf = int(baseline_frames)
    if bf < 1:
        base = 0.0
    else:
        bf = int(min(bf, int(y.size)))
        base = float(np.mean(y[:bf]))
    y0 = np.subtract(y, base)
    sig = float(sigma_frames)
    if (not np.isfinite(sig)) or sig <= 0.0:
        sig = 1.5
    ys = gaussian_filter1d(y0, sigma=sig, mode="nearest") + base
    if bool(clip_nonneg):
        ys = np.maximum(ys, 0.0)
    return ys.astype(np.float64)


def robust_median_positive(a, eps):
    a = a[np.isfinite(a) & (a > 0.0)]
    if int(a.size) < 20:
        return float(eps)
    v = float(np.median(a))
    return float(eps) if v <= 0.0 else v


def weight_from_aif(a):
    w = a.astype(np.float64).copy()
    w = np.subtract(w, float(np.min(w)))
    mx = float(np.max(w))
    if mx > 0.0:
        w = w / mx
    return 0.2 + 0.8 * np.clip(w, 0.0, 1.0)


def compute_peak_map(ctc, baseline_frames):
    t_len = int(ctc.shape[3])
    bf = int(min(int(baseline_frames), t_len))
    if bf < 1:
        base = np.zeros(ctc.shape[:3], dtype=np.float64)
    else:
        base = np.mean(ctc[..., :bf].astype(np.float64), axis=3)
    d = np.maximum(np.subtract(ctc.astype(np.float64), base[..., None]), 0.0)
    return np.max(d, axis=3).astype(np.float64)


def choose_peak_threshold(peak, mask):
    v = peak[mask]
    v = v[np.isfinite(v)]
    if int(v.size) < 100:
        return 0.0
    p20 = float(np.percentile(v, 20.0))
    p80 = float(np.percentile(v, 80.0))
    span = float(max(float(np.subtract(p80, p20)), 1.0 / 1e12))
    return float(max(p20 + 0.02 * span, 0.0))


def positive_excursion(x, baseline_frames):
    """Baseline-subtracted, non-negative curves; x has shape (n, t)."""
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float64)
    bf = int(min(int(baseline_frames), int(x.shape[1])))
    base = np.mean(x[:, :bf], axis=1, keepdims=True) if bf >= 1 else np.zeros((x.shape[0], 1))
    return np.maximum(np.subtract(x, base), 0.0)


def voxel_proxy_from_ctc(ctc_vox_t, dt, baseline_frames, eps):
    x = np.where(np.isfinite(ctc_vox_t), ctc_vox_t, 0.0).astype(np.float64)
    n = int(x.shape[0])
    bf = int(min(int(baseline_frames), int(x.shape[1])))
    base = np.zeros((n, 1), dtype=np.float64) if bf < 1 else np.mean(x[:, :bf], axis=1, keepdims=True).astype(np.float64)
    d = np.maximum(np.subtract(x, base), 0.0)
    peak = np.max(d, axis=1).astype(np.float64)
    ttp_sec = np.argmax(d, axis=1).astype(np.float64) * float(dt)
    proxy = np.divide(peak, ttp_sec + float(max(float(dt), float(eps))))
    proxy = np.where(np.isfinite(proxy), proxy, 0.0)
    return proxy.astype(np.float64), ttp_sec.astype(np.float64)


def voxel_auc_from_ctc(ctc_vox_t, dt, baseline_frames):
    """[C3] Voxel CBV texture: area under the positive excursion (indicator dilution)."""
    return np.sum(positive_excursion(ctc_vox_t, baseline_frames), axis=1) * float(dt)


def compute_features_for_kmeans(ctc4, coords3, dt, baseline_frames):
    eps = 1.0 / 1e12
    x = ctc4.astype(np.float64)
    n = int(x.shape[0])
    bf = int(min(int(baseline_frames), int(x.shape[1])))
    base = np.zeros((n, 1), dtype=np.float64) if bf < 1 else np.mean(x[:, :bf], axis=1, keepdims=True).astype(np.float64)
    d = np.maximum(np.subtract(x, base), 0.0)
    peak = np.max(d, axis=1).astype(np.float64)
    ttp_sec = np.argmax(d, axis=1).astype(np.float64) * float(dt)
    auc = np.sum(d, axis=1).astype(np.float64) * float(dt)
    cx = coords3[:, 0].astype(np.float64)
    cy = coords3[:, 1].astype(np.float64)
    cz = coords3[:, 2].astype(np.float64)
    cx = cx / float(max(float(np.max(cx)), 1.0))
    cy = cy / float(max(float(np.max(cy)), 1.0))
    cz = cz / float(max(float(np.max(cz)), 1.0))
    feats = np.stack([peak, ttp_sec, auc, cx, cy, cz], axis=1).astype(np.float64)
    feats = np.where(np.isfinite(feats), feats, 0.0)
    med = np.median(feats, axis=0, keepdims=True)
    mad = np.median(np.abs(np.subtract(feats, med)), axis=0, keepdims=True)
    scale = np.maximum(mad * 1.4826, eps)
    return np.divide(np.subtract(feats, med), scale).astype(np.float32)


def kmeans_numpy(x, k, seed, max_iter):
    rs = np.random.RandomState(int(seed))
    n = int(x.shape[0])
    if k < 2:
        return np.zeros(n, dtype=np.int32), np.mean(x, axis=0, keepdims=True).astype(np.float32)
    cent = x[rs.choice(n, size=int(k), replace=False)].astype(np.float32).copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(int(max_iter)):
        old = labels.copy()
        best_dist = np.full(n, np.inf, dtype=np.float32)
        best_lab = np.zeros(n, dtype=np.int32)
        for j in range(int(k)):
            diff = np.subtract(x, cent[j])
            dist = np.sum(diff * diff, axis=1).astype(np.float32)
            take = dist < best_dist
            best_dist[take] = dist[take]
            best_lab[take] = int(j)
        labels = best_lab
        for j in range(int(k)):
            sel = labels == int(j)
            if int(np.sum(sel)) < 1:
                cent[j] = x[int(rs.randint(0, n))]
            else:
                cent[j] = np.mean(x[sel], axis=0).astype(np.float32)
        if int(np.sum(labels != old)) == 0:
            break
    return labels.astype(np.int32), cent


def merge_small_clusters(labels, cent, min_vox):
    k = int(cent.shape[0])
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    large = np.where(counts >= int(min_vox))[0].astype(np.int32)
    if int(large.size) < 1:
        return np.zeros(labels.shape[0], dtype=np.int32), cent[:1].copy()
    lab2 = labels.copy()
    large_set = set(int(x) for x in large.tolist())
    for j in range(k):
        if j in large_set:
            continue
        diff = np.subtract(cent[large].astype(np.float64), cent[j].astype(np.float64)[None, :])
        j2 = int(large[int(np.argmin(np.sum(diff * diff, axis=1)))])
        lab2[lab2 == j] = j2
    used = np.unique(lab2).astype(np.int32)
    remap = {int(u): i for i, u in enumerate(used.tolist())}
    out = np.zeros_like(lab2)
    for u in used.tolist():
        out[lab2 == int(u)] = remap[int(u)]
    cent2 = np.zeros((int(used.size), int(cent.shape[1])), dtype=np.float32)
    for u in used.tolist():
        cent2[remap[int(u)]] = cent[int(u)]
    return out.astype(np.int32), cent2


def fill_nearest(vol3, inside_mask3, valid_mask3):
    holes = inside_mask3 & np.logical_not(valid_mask3)
    if int(np.sum(holes)) < 1 or int(np.sum(valid_mask3)) < 1:
        return vol3
    _, inds = distance_transform_edt(np.logical_not(valid_mask3), return_indices=True)
    out = vol3.copy()
    hx, hy, hz = np.where(holes)
    out[hx, hy, hz] = vol3[inds[0][hx, hy, hz], inds[1][hx, hy, hz], inds[2][hx, hy, hz]]
    return out
