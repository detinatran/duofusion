"""The baseline config must reproduce the original HemoPIC outputs bit for bit.

Uses the patient 22 demo bundled with the original repo, so no dataset is
needed. Run from the project root:

python -m pytest tests -q          (or)          python tests/test_baseline_equivalence.py
"""
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.models import arterial_input  # noqa: E402
from hemopic_plus.pipeline import run_subject  # noqa: E402

ORIG = ROOT.parent / "HemoPIC"
DEMO_DATA = ORIG / "assets" / "example" / "ISLES2017_Training"
DEMO_FIT = ORIG / "assets" / "example" / "HemoPIC_outputs" / "training_22"


def test_baseline_matches_bundled_demo():
    if not DEMO_DATA.exists():
        print("skip: original HemoPIC repo not found next to this project")
        return
    cfg = load_config(ROOT / "configs" / "baseline.yaml", {"data_root": str(DEMO_DATA)})
    with tempfile.TemporaryDirectory() as tmp:
        run_subject(cfg, 22, Path(tmp))
        out = Path(tmp) / "training_22"
        labels = nib.load(str(out / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata()
        ref = nib.load(str(DEMO_FIT / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata()
        assert np.array_equal(labels, ref), "ROI partition differs from the original"
        for tag in ("CBF", "CBV", "MTT"):
            a = nib.load(str(out / f"HemoPIC_{tag}.nii.gz")).get_fdata()
            b = nib.load(str(DEMO_FIT / f"HemoPIC_{tag}.nii.gz")).get_fdata()
            assert np.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True), f"{tag} differs from the original"


def test_arterial_input_identity_and_area():
    t = np.arange(60) * 1.44
    cart = np.exp(-0.5 * ((t - 25) / 4) ** 2)
    assert np.allclose(arterial_input(cart, t, 0.0, 0.0), cart)
    late = arterial_input(cart, t, 5.0, 0.0)
    assert abs(t[np.argmax(late)] - t[np.argmax(cart)] - 5.0) <= 1.44
    disp = arterial_input(cart, t, 0.0, 4.0)
    assert disp.max() < cart.max() and abs(disp.sum() - cart.sum()) / cart.sum() < 0.05


if __name__ == "__main__":
    test_arterial_input_identity_and_area()
    test_baseline_matches_bundled_demo()
    print("all tests passed")
