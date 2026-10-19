"""Reference CBF per voxel, the denominator of the relative CBF used to delineate the core.

Every reference uses only the admission CTP and NCCT; the follow-up infarct is never read.
Healthy tissue is tissue with 0 <= Tmax < 4 s, hypoperfused tissue has Tmax > 6 s.

  brain_median      median CBF of the brain (HemoPIC+'s normalisation)
  contra_hemi       median CBF of healthy tissue in the unaffected hemisphere (RAPID-like)
  contra_tissue     as contra_hemi, separately for cortex, white matter, deep grey matter, ...
  healthy_tissue    median CBF of healthy tissue of the same class in both hemispheres
  mirror            smoothed CBF of the contralateral homologue (falls back to contra_tissue
                    where the homologue is outside the field of view or hypoperfused)
  normative         gradient-boosted regression of log CBF on anatomy, the patient's healthy
                    CBF level and the homologue, trained on healthy voxels of other patients
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import HistGradientBoostingRegressor

from .labels import TISSUE_CODE, side_free_label

HEALTHY_TMAX, HYPO_TMAX = 4.0, 6.0


def nan_gaussian(vol, sigma_mm, spacing):
    """Gaussian smoothing that ignores NaN (normalised convolution)."""
    sig = [sigma_mm / float(s) for s in spacing]
    ok = np.isfinite(vol)
    num = gaussian_filter(np.where(ok, vol, 0.0), sig)
    den = gaussian_filter(ok.astype(np.float64), sig)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.2] = np.nan
    return out


def spacing_of(affine):
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))


def masks(s, cbf, tmax):
    """valid (tissue with a positive, finite CBF), healthy and hypoperfused voxels.
    Negative Tmax (vendor artefacts, a few % of tissue voxels) counts as neither."""
    valid = (s.tissue > 0) & np.isfinite(cbf) & (cbf > 0)
    t = np.where(np.isfinite(tmax), tmax, np.nan)
    with np.errstate(invalid="ignore"):
        return valid, valid & (t >= 0) & (t < HEALTHY_TMAX), valid & (t > HYPO_TMAX)


def affected_side(s, hypo):
    """Hemisphere (1 left, 2 right) with the larger hypoperfused volume, 0 if none."""
    left, right = int(np.sum(hypo & (s.hemi == 1))), int(np.sum(hypo & (s.hemi == 2)))
    if left == right == 0:
        return 0
    return 1 if left > right else 2


def _median(x, default=np.nan):
    return float(np.median(x)) if x.size else default


def per_tissue(s, cbf, sel, fallback):
    """Volume holding, at every tissue voxel, the median CBF of sel within its tissue class."""
    out = np.full(cbf.shape, np.nan)
    for code in TISSUE_CODE.values():
        here = s.tissue == code
        out[here] = _median(cbf[sel & here], fallback)
    return out


def classic_references(s, cbf, tmax, mirror=None, sigma_mm=4.0):
    """The rule-based references as a dict name -> volume (NaN outside tissue)."""
    valid, healthy, hypo = masks(s, cbf, tmax)
    side = affected_side(s, hypo)
    contra = (s.hemi == (3 - side)) if side else (s.hemi > 0)
    brain_med = _median(cbf[valid])
    healthy_med = _median(cbf[healthy], brain_med)
    contra_sel = healthy & contra if np.any(healthy & contra) else healthy
    refs = {
        "brain_median": np.where(valid, brain_med, np.nan),
        "contra_hemi": np.where(valid, _median(cbf[contra_sel], healthy_med), np.nan),
        "contra_tissue": np.where(valid, per_tissue(s, cbf, contra_sel, healthy_med), np.nan),
        "healthy_tissue": np.where(valid, per_tissue(s, cbf, healthy, healthy_med), np.nan),
    }
    if mirror is not None:
        sp = spacing_of(s.affine)
        smooth = nan_gaussian(np.where(valid, cbf, np.nan), sigma_mm, sp)
        m_cbf = mirror(smooth)
        m_tmax = mirror(np.where(np.isfinite(tmax), tmax, np.nan))
        ok = np.isfinite(m_cbf) & (m_cbf > 0) & ~(np.isfinite(m_tmax) & (m_tmax > HYPO_TMAX))
        refs["mirror"] = np.where(valid, np.where(ok, m_cbf, refs["contra_tissue"]), np.nan)
    return refs, {"affected_side": side, "brain_median": brain_med, "healthy_median": healthy_med}


# ----------------------------------------------------------------------------- normative

FEATURES = ("tissue", "label", "midline_mm", "ap_mm", "si_mm", "lvl_cortex", "lvl_wm", "lvl_deep",
            "mirror_logcbf", "mirror_tmax", "age")
CATEGORICAL = (0, 1)


def voxel_features(s, cbf, tmax, mirror=None, sigma_mm=4.0):
    """Feature matrix for the tissue voxels of one subject (rows in np.nonzero(valid) order).
    The voxel's own CBF and Tmax are deliberately not features."""
    valid, healthy, _ = masks(s, cbf, tmax)
    idx = np.nonzero(valid)
    ijk = np.stack(idx + (np.ones(idx[0].size),), axis=0)
    xyz = (np.asarray(s.affine) @ ijk)[:3].T                      # RAS world coordinates (mm)
    lat = s.hemi[idx]
    left_x = xyz[lat == 1, 0].mean() if np.any(lat == 1) else xyz[:, 0].mean()
    right_x = xyz[lat == 2, 0].mean() if np.any(lat == 2) else xyz[:, 0].mean()
    mid = 0.5 * (left_x + right_x)
    centre = xyz.mean(axis=0)
    lvl = {}
    for name in ("cortex", "wm", "deep"):
        sel = healthy & (s.tissue == TISSUE_CODE[name])
        lvl[name] = np.log(_median(cbf[sel], _median(cbf[healthy], _median(cbf[valid], 1.0))))
    if mirror is not None:
        smooth = nan_gaussian(np.where(valid, cbf, np.nan), sigma_mm, spacing_of(s.affine))
        with np.errstate(invalid="ignore", divide="ignore"):
            m_log = np.log(mirror(smooth))[idx]
        m_tmax = mirror(np.where(np.isfinite(tmax), tmax, np.nan))[idx]
    else:
        m_log = m_tmax = np.full(idx[0].size, np.nan)
    age = s.info.get("age")
    X = np.column_stack([
        s.tissue[idx].astype(np.float64),
        side_free_label(s.seg)[idx].astype(np.float64),
        np.abs(xyz[:, 0] - mid), xyz[:, 1] - centre[1], xyz[:, 2] - centre[2],
        np.full(idx[0].size, lvl["cortex"]), np.full(idx[0].size, lvl["wm"]), np.full(idx[0].size, lvl["deep"]),
        m_log, m_tmax,
        np.full(idx[0].size, float(age) if age is not None else np.nan),
    ])
    return X, idx, valid, healthy


def training_rows(s, cbf, tmax, mirror=None, max_rows=20000, sigma_mm=2.0, seed=0):
    """Healthy voxels of one training subject: features and log of lightly smoothed CBF."""
    X, idx, valid, healthy = voxel_features(s, cbf, tmax, mirror)
    target = nan_gaussian(np.where(valid, cbf, np.nan), sigma_mm, spacing_of(s.affine))[idx]
    keep = healthy[idx] & np.isfinite(target) & (target > 0)
    rows = np.nonzero(keep)[0]
    if rows.size > max_rows:
        rows = np.random.default_rng(seed + s.num).choice(rows, max_rows, replace=False)
    return X[rows], np.log(target[rows])


class NormativeReference:
    """log CBF ~ anatomy + patient level + homologue, fitted on healthy voxels."""

    def __init__(self, seed=0):
        self.seed = seed
        self.keep = None
        self.model = None

    def fit(self, X, y):
        # columns without at least two distinct finite values (e.g. age missing in a cohort) are dropped
        self.keep = [j for j in range(X.shape[1]) if np.unique(X[np.isfinite(X[:, j]), j]).size >= 2]
        cat = [self.keep.index(j) for j in CATEGORICAL if j in self.keep]
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                   min_samples_leaf=200, l2_regularization=1.0,
                                                   categorical_features=cat or None, random_state=self.seed)
        self.model.fit(X[:, self.keep], y)
        return self

    def predict_volume(self, s, cbf, tmax, mirror=None):
        X, idx, valid, _ = voxel_features(s, cbf, tmax, mirror)
        out = np.full(cbf.shape, np.nan)
        out[idx] = np.exp(self.model.predict(X[:, self.keep]))
        return out
