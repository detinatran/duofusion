"""Left-right mirroring about the patient's midsagittal plane.

The volume is flipped along the voxel axis closest to the left-right direction, and
the flipped image is registered rigidly to the original, so that the composition maps
every voxel to its contralateral homologue even when the head is shifted or tilted in
the field of view. Thin slabs (fewer than PLANAR_BELOW slices along the third voxel
axis, as in ISLES 2018) are registered in plane on their mean over slices, and the 2D
transform is applied to every slice.
"""
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PLANAR_BELOW = 12


def lr_axis(affine):
    """Voxel axis whose direction has the largest left-right (RAS x) component."""
    return int(np.argmax(np.abs(np.asarray(affine)[0, :3])))


def _to_sitk(vol, spacing):
    """numpy volume (2D or 3D, index order x, y[, z]) -> SimpleITK image."""
    arr = np.ascontiguousarray(np.transpose(vol)).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in spacing))
    return img


def _from_sitk(img):
    return np.transpose(sitk.GetArrayFromImage(img))


def _register(fixed, moving, mask, dim, seed):
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform() if dim == 3 else sitk.Euler2DTransform(),
                                             sitk.CenteredTransformInitializerFilter.MOMENTS)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetMetricFixedMask(sitk.Cast(mask, sitk.sitkUInt8))
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2, seed)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300, relaxationFactor=0.6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([2, 1])
    R.SetSmoothingSigmasPerLevel([1, 0])
    R.SetInitialTransform(init, inPlace=False)
    tx = R.Execute(fixed, moving)
    return tx, float(R.GetMetricValue())


class Mirror:
    """mirror(vol) returns, at every voxel, the value of vol at its contralateral homologue."""

    def __init__(self, affine, intensity, mask, seed=0, transform_file=None):
        self.axis = lr_axis(affine)
        self.spacing = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))
        self.shape = intensity.shape
        self.planar = intensity.shape[2] < PLANAR_BELOW and self.axis != 2
        img = np.where(mask, intensity, 0.0)
        if self.planar:
            img = img.mean(axis=2)
            mask = mask.any(axis=2)
        sp = self.spacing[:2] if self.planar else self.spacing
        self._ref = _to_sitk(img, sp)
        if transform_file is not None and Path(transform_file).exists():
            self.transform = sitk.ReadTransform(str(transform_file))
            self.metric = float("nan")
            return
        moving = _to_sitk(np.flip(img, self.axis), sp)
        self.transform, self.metric = _register(self._ref, moving, _to_sitk(mask.astype(np.float32), sp),
                                                2 if self.planar else 3, seed)
        if transform_file is not None:
            sitk.WriteTransform(self.transform, str(transform_file))

    def _resample(self, vol, interp, fill):
        sp = self.spacing[:2] if self.planar else self.spacing
        moving = _to_sitk(np.flip(vol, self.axis), sp)
        return _from_sitk(sitk.Resample(moving, self._ref, self.transform, interp, fill))

    def __call__(self, vol, nearest=False, fill=np.nan):
        vol = np.asarray(vol, np.float32)
        interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
        f0 = float(fill) if np.isfinite(fill) else 0.0
        if self.planar:
            out = np.stack([self._resample(vol[..., k], interp, f0) for k in range(vol.shape[2])], axis=2)
            inside = self._resample(np.ones(vol.shape[:2], np.float32), sitk.sitkNearestNeighbor, 0.0)[..., None] > 0
        else:
            out = self._resample(vol, interp, f0)
            inside = self._resample(np.ones(vol.shape, np.float32), sitk.sitkNearestNeighbor, 0.0) > 0
        if not np.isfinite(fill):
            out = np.where(np.broadcast_to(inside, out.shape), out, np.nan)
        return out

    def side(self, mask):
        """1/2 for the two sides of the midsagittal plane (1 = lower index along the left-right
        axis than its homologue), 0 outside mask or where the homologue is outside the image."""
        idx = np.indices(self.shape, dtype=np.float32)[self.axis]
        hom = self(idx)
        out = np.zeros(self.shape, np.uint8)
        ok = mask & np.isfinite(hom)
        out[ok & (idx < hom)] = 1
        out[ok & (idx >= hom)] = 2
        return out

    def quality(self, hemi):
        """Fraction of lateralised voxels whose mirror lies in the opposite hemisphere."""
        m = self(hemi.astype(np.float32), nearest=True, fill=0.0)
        lat = hemi > 0
        if not np.any(lat):
            return float("nan")
        return float(np.mean((m[lat] > 0) & (m[lat] != hemi[lat])))
