"""The PyTorch ROI model must reproduce the numpy fit (curves and objective).

Needs fitted outputs, e.g. python scripts/run_cohort.py configs/full.yaml --patients 22
python tests/test_uq_torch.py [outputs/full/training_22 ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hemopic_plus.uq.data import load_subject  # noqa: E402
from hemopic_plus.uq.torch_model import objective, simulate  # noqa: E402


def check(fit_dir):
    b = load_subject(fit_dir)
    with torch.no_grad():
        _, c_fit = simulate(b, b.theta_fit)
        obj, data, num, den = objective(b, b.theta_fit)
    c_np = b.c_fit_numpy
    err = float(np.max(np.abs(c_fit.numpy() - c_np)) / max(np.max(np.abs(c_np)), 1e-12))
    w = b.w.numpy()
    data_np = np.mean(w * (c_np - b.c_obs.numpy()) ** 2, axis=1) / (np.mean(w * b.c_obs.numpy() ** 2, axis=1) + 1e-12)
    derr = float(np.max(np.abs(data.numpy() - data_np)))
    assert err < 1e-6, f"{fit_dir}: curve mismatch {err:.2e}"
    assert derr < 1e-8, f"{fit_dir}: objective mismatch {derr:.2e}"
    return err, derr, len(b.roi_ids), b.P


if __name__ == "__main__":
    dirs = sys.argv[1:] or [str(p) for p in sorted((ROOT / "outputs").glob("*/training_22"))
                            if (p / "fit_curves.npz").exists() and "__" not in p.parent.name]
    if not dirs:
        print("skip: no fitted outputs found")
    for d in dirs:
        print(d, "max rel curve err %.2e, objective err %.2e, %d ROIs, P=%d" % check(d))
    print("all tests passed")
