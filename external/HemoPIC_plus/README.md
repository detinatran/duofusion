# HemoPIC+ (vendored)

The physics-based perfusion twin used by DuoFusion. HemoPIC+ extends
[HemoPIC](https://github.com/jhuldr/HemoPIC) (MICCAI 2026; MIT licence, see `LICENSE_HemoPIC`).
For every region of interest it fits a Windkessel flow model and a transport tissue model to the
tracer curve, and a voxel-level fit (`hemopic_plus/npde/voxel.py`) gives each voxel its own volume,
inflow concentration and delay. DuoFusion uses the ROI fits, the voxel fits, the ROI posterior
(`scripts/run_uq.py`) and the indicator-dilution CBV.

Entry points used by DuoFusion:

| script | role |
|---|---|
| `scripts/prepare_isles24.py` | ISLES'24 CT perfusion in the CTP grid |
| `scripts/smooth_external.py` | spatially smoothed copy of a cohort |
| `scripts/run_cohort.py configs/isles24_full_aif.yaml` | ROI fits with the automatic AIF |
| `scripts/run_cohort.py configs/isles24_full_aif_idvox_osvd_nc.yaml` | indicator-dilution CBV |
| `scripts/run_uq.py` | ROI posterior (tempered Laplace) |
| `scripts/eval_external.py` | voxel fits and lesion AUC against oSVD |

The package keeps its working name, `hemopic_plus`. Requirements: `requirements.txt`.
